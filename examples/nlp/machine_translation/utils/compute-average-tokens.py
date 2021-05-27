#!/usr/bin/env python
"""
Prints the evrage token number from a tarred dataset.
"""

import pickle
import glob
import numpy as np
import sys

# files = glob.glob("*.pkl")
files = sys.argv[1:]
all_len = []

for fn in files:
    d = pickle.load(open(fn, "rb"))
    all_len.extend(np.where(d['tgt'] == 3)[1])

print(len(all_len))
print(np.mean(all_len))
