"""
Data pre-processing: encode labels (age, diagnosis, patient id) and crop data.
"""

import argparse
import os
import functools
import math
import linecache
import glob
import scipy.io
import numpy as np

from multiprocessing import Pool

from fairseq_signals.data.ecg import ecg_utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="DIR",
                       help="root directory containing mat files to pre-process")
    parser.add_argument("--meta-dir",
                       help="directory containing metadata for labeling (weights.csv)")
    parser.add_argument(
        "--subset",
        default="WFDB_CPSC2018, WFDB_CPSC2018_2, WFDB_Ga, WFDB_PTBXL, WFDB_ChapmanShaoxing, WFDB_Ningbo",
        type=str,
        help="comma separated list of sub-directories of data subsets to be preprocessed, "
        "each of which is labeled seperately (e.g. WFDB_CPSC2018, WFDB_CPSC2018_2, ...)"
    )
    parser.add_argument("--dest", type=str, metavar="DIR",
                       help="output directory")
    parser.add_argument(
        "--predir", default=".", type=str, metavar="DIR", help="if set, create sub-root directory in --dest"
    )
    parser.add_argument("--ext", default="mat", type=str, metavar="EXT",
                       help="extension to look for")
    parser.add_argument("--sec", default=5, type=int,
                       help="seconds to repeatedly crop to")
    parser.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    return parser

def main(args):
    if not args.meta_dir:
        args.meta_dir = args.root

    meta_path = os.path.join(os.path.realpath(args.meta_dir), "weights.csv")

    subset = args.subset.replace(' ','').split(',')
    for s in subset:
        if not os.path.exists(os.path.join(args.dest, args.predir, s.lstrip("WFDB_"))):
            os.makedirs(os.path.join(args.dest, args.predir, s.lstrip("WFDB_")))

        try:
            classes, _ = ecg_utils.get_physionet_weights(meta_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "cannot find the metadata file for labeling (weights.csv)"
                "please ensure that files are located in --meta-dir "
                "or download from https://github.com/physionetchallenges/evaluation-2021."
                f"--meta-dir: {meta_path}"
            )

        dir_path = os.path.realpath(os.path.join(args.root, s))
        search_path = os.path.join(dir_path, "**/*." + args.ext)

        fnames = list(glob.iglob(search_path, recursive=True))
        chunk_size = math.ceil(len(fnames) / args.workers)

        file_chunks = [fnames[i:i+chunk_size] for i in range(0, len(fnames), chunk_size)]

        func = functools.partial(
            preprocess,
            args,
            classes,
            os.path.join(args.dest, args.predir, s.lstrip("WFDB_"))
        )
        pool = Pool(processes = args.workers)
        pool.map(func, file_chunks)
        pool.close()
        pool.join()

def preprocess(args, classes, dest_path, fnames):
    for fname in fnames:
        fname = fname[:-(len(args.ext)+1)]

        y = set(linecache.getline(fname + '.hea', 16).replace(',',' ').split()[1:])
        label = np.zeros(len(classes), dtype=bool)
        for i, x in enumerate(classes):
            if x & y:
                label[i] = 1

        try:
            age = int(linecache.getline(fname + '.hea', 14).split()[1])
        except ValueError:
            age = 0
        sex = 0 if linecache.getline(fname + '.hea', 15).split()[1] == "Male" else 1

        sample_rate = int(linecache.getline(fname + '.hea', 1).split()[2])

        # 500hz is expected
        if sample_rate != 500:
            continue

        record = scipy.io.loadmat(fname)

        length = record['val'].shape[-1]

        for i, seg in enumerate(range(0, length, int(args.sec * sample_rate))):
            data = {}
            data['age'] = age
            data['sex'] = sex
            data['label'] = label
            data['patient_id'] = os.path.basename(fname)
            data['curr_sample_rate'] = sample_rate
            if seg + args.sec * sample_rate <= length:
                data['feats'] = record['val'][:, seg: int(seg + args.sec * sample_rate)]
                scipy.io.savemat(os.path.join(dest_path, os.path.basename(fname) + f"_{i}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)