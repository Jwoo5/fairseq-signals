"""
Data pre-processing: encode labels (patient id) and crop data
"""

import argparse
import os
import functools
import math
import linecache
import glob
import wfdb
import scipy.io
import numpy as np

from multiprocessing import Pool

from fairseq_signals.data.ecg import ecg_utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="root directory containing dat files to pre-process"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--ext", default="dat", type=str, metavar="EXT",
        help="extension to look for"
    )
    parser.add_argument(
        "--sec", default=5, type=int,
        help="seconds to repeatedly crop to"
    )
    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    fnames = list(glob.iglob(search_path, recursive=True))

    for fname in fnames:
        fname = os.path.splitext(fname)[0]

        # take a raw signal (index 1 is for a filtered signal)
        record = wfdb.rdrecord(fname).__dict__['p_signal'][:,0]
        annot = wfdb.rdann(fname, 'atr')

        sample_rate = annot.__dict__['fs']

        # XXX add lead info?

        # 500hz is expected
        if sample_rate != 500:
            continue

        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        length = record.shape[-1]

        pid = fname.split('/')[-2][-2:]
        basename = os.path.basename(fname)
        record = record.astype(np.float32)

        start = np.random.randint(length - (args.sec * sample_rate))

        data = {}
        data['patient_id'] = pid
        data['curr_sample_rate'] = sample_rate
        data['feats'] = record[start: start + (args.sec * sample_rate)]
        scipy.io.savemat(os.path.join(dest_path, f"{pid}_{basename}.mat"), data)

        # for i, seg in enumerate(range(0, length, int(args.sec * sample_rate))):
        #     data = {}
        #     data['patient_id'] = pid
        #     data['curr_sample_rate'] = sample_rate
        #     if seg + args.sec * sample_rate <= length:
        #         data['feats'] = record[seg: int(seg + args.sec * sample_rate)]
        #         scipy.io.savemat(os.path.join(dest_path, f"{pid}_{basename}_{i}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)