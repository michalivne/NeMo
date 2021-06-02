#!/usr/bin/env python
"""
Saves a .nemo file given a checkpoint
"""

import sys
import os
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTMIMModel, MTEncDecModel

print("{script} <seq2seq or mim> <.ckpt input file> <.nemo output file>".format(
    script=os.path.basename(sys.argv[0])))

model_type = sys.argv[1]
checkpoint_fname = sys.argv[2]
nemo_fname = sys.argv[3]

if model_type == "mim":
    model_cls = MTMIMModel
elif model_type == "seq2seq":
    model_cls = MTEncDecModel
else:
    raise ValueError("Expected model_type in ['mim', 'seq2seq']")

if not checkpoint_fname.endswith(".ckpt"):
    checkpoint_fname += ".ckpt"

if not nemo_fname.endswith(".nemo"):
    nemo_fname += ".nemo"


print(f"Loading checkpoint_fname = {checkpoint_fname}")
model = model_cls.load_from_checkpoint(checkpoint_fname)

print(f"Saving nemo_fname = {nemo_fname}")
model.save_to(nemo_fname)
