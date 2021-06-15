# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Given NMT model's .nemo file, this script can be used to translate text.
USAGE Example:
1. Obtain text file in src language. You can use sacrebleu to obtain standard test sets like so:
    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src
2. Translate:
    python nmt_transformer_infer.py --model=[Path to .nemo file] --srctext=wmt14-de-en.src --tgtout=wmt14-de-en.pre
"""


from argparse import ArgumentParser

import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging

import time
import datetime


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--srctext", type=str, required=True, help="")
    parser.add_argument("--tgtout", type=str, required=True, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--beam_size", type=int, default=4, help="")
    parser.add_argument("--len_pen", type=float, default=0.6, help="")
    parser.add_argument("--max_delta_length", type=int, default=5, help="")
    parser.add_argument("--target_lang", type=str, default=None, help="")
    parser.add_argument("--source_lang", type=str, default=None, help="")
    parser.add_argument("--fixed_len_penaly", type=float, default=-1, help="")
    # If given, append a line with current execution time to timeout file name
    parser.add_argument("--timeout", type=str, default="", help="")
    # if > 0 will profile the specified amount of batches
    parser.add_argument("--profile_batches", type=int, default=-1, help="")
    # If given, will save profiler output in chrome tracing format
    parser.add_argument("--profout", type=str, default="", help="")


    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=args.model)
        src_text = []
        tgt_text = []
    else:
        raise NotImplemented(f"Only support .nemo files, but got: {args.model}")

    model.beam_search.beam_size = args.beam_size
    model.beam_search.len_pen = args.len_pen
    model.beam_search.max_delta_length = args.max_delta_length
    if args.fixed_len_penaly > 0:
        model.beam_search.fixed_len_penaly = args.fixed_len_penaly

    if torch.cuda.is_available():
        model = model.cuda()

    logging.info(f"Translating: {args.srctext}")

    total_time = 0

    profile_enable = (args.profile_batches > 0)

    count = 0
    with open(args.srctext, 'r') as src_f:
        with torch.autograd.profiler.profile(
            enabled=profile_enable,
            use_cuda=torch.cuda.is_available(),
            record_shapes=False,
            with_stack=True,
            ) as prof:
            for line in src_f:
                src_text.append(line.strip())
                if len(src_text) == args.batch_size:
                    t0 = time.time()
                    res = model.translate(text=src_text, source_lang=args.source_lang, target_lang=args.target_lang)
                    t1 = time.time()
                    total_time = total_time + t1 - t0
                    if len(res) != len(src_text):
                        print(len(res))
                        print(len(src_text))
                        print(res)
                        print(src_text)
                    tgt_text += res
                    src_text = []

                    if profile_enable:
                        args.profile_batches -= 1
                        if args.profile_batches <= 0:
                            break

                count += 1
                # if count % 300 == 0:
                #    print(f"Translated {count} sentences")
            if len(src_text) > 0:
                t0 = time.time()
                tgt_text += model.translate(text=src_text, source_lang=args.source_lang, target_lang=args.target_lang)
                t1 = time.time()
                total_time = total_time + t1 - t0

    logging.info("Translation time: {total_time} [{ftotal_time}]".format(
        total_time=total_time,
        ftotal_time=str(datetime.timedelta(seconds=total_time)),
    ))

    if args.timeout:
        with open(args.timeout, "a") as fh:
            fh.write(f"{total_time}\n")

    with open(args.tgtout, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")

    if profile_enable:
        # save trace if output file is given
        if args.profout:
            prof.export_chrome_trace(args.profout)

        # print trace
        logging.info(prof.key_averages().table(sort_by="self_cpu_time_total"))

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
