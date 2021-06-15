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
Given NMT model's .nemo file, this script can be used to measure the speed of
inference for various batch sizes, and sequence lengths.
"""


from argparse import ArgumentParser

import torch
import numpy as np
import pprint

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging

import time
import datetime


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--input_seq", type=str, default="Para preparar el terreno de la Segunda Guerra Mundial, permitieron que los banqueros y políticos crearan, de modo encubierto, una situación conflictiva a raíz de las enormes reparaciones de guerra impuestas sobre Alemania, con lo que se crearon las condiciones óptimas para la radicalización de las masas empobrecidas. Posteriormente, les bastó con servirles a los alemanes la solución del Führer, que era lo suficientemente fuerte y simple y señalaba a culpables, y construir después una Checoslovaquia compuesta por varias nacionalidades con una fuerte minoría alemana que debía representar y representó tan bien la función de la quinta columna, que prendió la llama y provocó el estallido de la guerra.", help="")
    parser.add_argument("--batch_size", type=int, default=256, nargs='+', help="")
    parser.add_argument("--seq_len", type=int, default=16, nargs='+', help="")
    parser.add_argument("--beam_size", type=int, default=1, nargs='+', help="")
    parser.add_argument("--batches", type=int, default=100, help="")
    parser.add_argument("--len_pen", type=float, default=0.6, help="")
    parser.add_argument("--max_delta_length", type=int, default=5, help="")
    parser.add_argument("--target_lang", type=str, default=None, help="")
    parser.add_argument("--source_lang", type=str, default=None, help="")
    # If given, will save results
    parser.add_argument("--results_out", type=str, default="", help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=args.model)
        src_text = []
    else:
        raise NotImplemented(f"Only support .nemo files, but got: {args.model}")

    model.beam_search.len_pen = args.len_pen
    model.beam_search.max_delta_length = args.max_delta_length

    if torch.cuda.is_available():
        model = model.cuda()

    input_seq = args.input_seq.strip()
    input_seq_words = input_seq.split(' ') * 2
    N = len(input_seq_words)
    results_dict = dict()

    for beam_size in args.beam_size:
        model.beam_search.beam_size = beam_size

        for batch_size in args.batch_size:
            for seq_len in args.seq_len:
                name = f"beam={beam_size}_batch={batch_size}_seq_len={seq_len}"
                print(name)
                # build a batch
                src_text = []
                I0 = N - seq_len
                for b in range(batch_size):
                    i0 = np.random.randint(I0)
                    i1 = i0 + seq_len
                    src_text.append(' '.join(input_seq_words[i0:i1]))

                # repeat each batch multiple times
                cur_time = []
                for i in range(args.batches):
                    t0 = time.time()
                    res = model.translate(text=src_text, source_lang=args.source_lang, target_lang=args.target_lang)
                    t1 = time.time()
                    cur_time.append(t1 - t0)

                results_dict[name] = np.mean(cur_time)
                print("{mean} +/- {std}".format(
                    mean=np.mean(cur_time),
                    std=np.std(cur_time),
                ))

    # print results
    fresults = pprint.pformat(results_dict)
    logger.info(fresults)

    if args.results_out:
        with open(args.results_out, "w") as fh:
            fh.write(f"{fresults}\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
