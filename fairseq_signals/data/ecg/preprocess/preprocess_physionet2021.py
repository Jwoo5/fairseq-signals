"""
Data pre-processing: encode labels (age, diagnosis, patient id) and crop data.
"""

import argparse
import os
import scipy.io
import linecache
import numpy as np
import numpy.ma as ma
import glob
import pandas as pd
import math
import functools
from multiprocessing import Pool

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="DIR",
                       help="root directory containing mat files to pre-process")
    #TODO if not given, should infer labels on the fly
    parser.add_argument("--meta-dir",
                       help="directory containing metadata file (dx_mapping_(un)scored.csv)")
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

    meta_path = os.path.realpath(args.meta_dir)
    try:
        dx_codes = pd.read_csv(os.path.join(meta_path, "dx_mapping_scored.csv"))
        dx_codes = dx_codes.append(
                pd.read_csv(os.path.join(meta_path, "dx_mapping_unscored.csv")), ignore_index = True
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            "cannot find the metadata files for labeling (dx_mapping_(un)scored.csv)"
            "please ensure that files are located in --meta-dir "
            "or download from https://github.com/physionetchallenges/evaluation-2021."
            f"--meta-dir: {meta_path}"
        )

    def convert_to_column_names(subset):
        subset = subset.lstrip("WFDB_")
        if subset == "CPSC2018":
            return "CPSC"
        elif subset == "CPSC2018_2":
            return "CPSC_Extra"
        elif subset == "Ga":
            return "Georgia"
        elif subset == "PTBXL":
            return "PTB_XL"
        elif subset == "ChapmanShaoxing":
            return "Chapman_Shaoxing"
        elif subset == "Ningbo":
            return "Ningbo"
        else:
            raise NotImplementedError()

    subset = args.subset.replace(' ','').split(',')

    for s in subset:
        if not os.path.exists(os.path.join(args.dest, args.predir, s.lstrip("WFDB_"))):
            os.makedirs(os.path.join(args.dest, args.predir, s.lstrip("WFDB_")))

        codes = dx_codes[dx_codes[convert_to_column_names(s)] > 0]["SNOMEDCTCode"].to_numpy().astype(str)

        dir_path = os.path.realpath(os.path.join(args.root, s))
        search_path = os.path.join(dir_path, "**/*." + args.ext)

        fnames = list(glob.iglob(search_path, recursive=True))
        chunk_size = math.ceil(len(fnames) / args.workers)

        file_chunks = [fnames[i:i+chunk_size] for i in range(0, len(fnames), chunk_size)]

        func = functools.partial(
            preprocess,
            args,
            codes,
            os.path.join(args.dest, args.predir, s.lstrip("WFDB_"))
        )
        pool = Pool(processes = args.workers)
        pool.map(func, file_chunks)
        pool.close()
        pool.join()

def preprocess(args, dx_codes, dest_path, fnames):
    for fname in fnames:
        fname = fname[:-(len(args.ext)+1)]
        label = linecache.getline(fname + '.hea', 16).replace(',',' ').split()[1:]
        mask = np.zeros(len(dx_codes))
        for each in label:
            mask = np.logical_or(mask, np.where(dx_codes == each, True, False))
        label = np.zeros(len(dx_codes), dtype = np.int16)
        label = ma.masked_array(label, mask).filled(fill_value=1)

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

        num_segs = length // int(args.sec * sample_rate)
        for i, seg in enumerate(range(0, length, int(args.sec * sample_rate))):
            data = {}
            data['age'] = age
            data['sex'] = sex
            data['label'] = label
            data['patient_id'] = os.path.basename(fname)
            data['curr_sample_rate'] = sample_rate
            # data['segment'] = 0 if i % 2 == 0 else 1
            data['num_segs'] = num_segs
            if seg + args.sec * sample_rate <= length:
                data['feats'] = record['val'][:, seg: int(seg + args.sec * sample_rate)]
                scipy.io.savemat(os.path.join(dest_path, os.path.basename(fname) + f"_{i}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)