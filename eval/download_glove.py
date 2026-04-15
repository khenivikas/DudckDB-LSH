#!/usr/bin/env python3
"""
Convert GloVe-200 HDF5 (from ANN-Benchmarks) to fvecs/ivecs format.

Place glove-200-angular.hdf5 in eval/data/glove/ then run:
    python download_glove.py
"""

import os
import struct

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOVE_DIR = os.path.join(SCRIPT_DIR, "data", "glove")
HDF5_PATH = os.path.join(GLOVE_DIR, "glove-200-angular.hdf5")


def write_fvecs(path, vecs):
    with open(path, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))
    print(f"  Wrote {len(vecs)} vectors to {path}")


def write_ivecs(path, vecs):
    with open(path, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}i", *vec))
    print(f"  Wrote {len(vecs)} vectors to {path}")


def main():
    if not os.path.exists(HDF5_PATH):
        print(f"ERROR: {HDF5_PATH} not found.")
        print("Place glove-200-angular.hdf5 in eval/data/glove/")
        return

    print("Converting HDF5 to fvecs/ivecs ...")
    with h5py.File(HDF5_PATH, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)

    print(f"  train: {train.shape}, test: {test.shape}, neighbors: {neighbors.shape}")

    write_fvecs(os.path.join(GLOVE_DIR, "glove_base.fvecs"), train)
    write_fvecs(os.path.join(GLOVE_DIR, "glove_query.fvecs"), test)
    write_ivecs(os.path.join(GLOVE_DIR, "glove_groundtruth.ivecs"), neighbors)
    print("Done.")


if __name__ == "__main__":
    main()
