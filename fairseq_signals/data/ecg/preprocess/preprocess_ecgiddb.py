"""
Data pre-processing: encode labels (patient id) and crop data
"""

import argparse
import os
import glob
import wfdb
import scipy.io
import numpy as np

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
        help="seconds to randomly crop to"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    fnames = list(glob.iglob(search_path, recursive=True))

    pid_table = dict()

    np.random.seed(args.seed)
    i = 0
    for fname in fnames:
        fname = os.path.splitext(fname)[0]

        # take a filtered signal (index 1 is for a filtered signal)
        record = wfdb.rdrecord(fname).__dict__['p_signal'][:,1]
        annot = wfdb.rdann(fname, 'atr')

        sample_rate = annot.__dict__['fs']

        # 500hz is expected
        if sample_rate != 500:
            continue

        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        length = record.shape[-1]

        pid = fname.split('/')[-2][-2:]
        if pid not in pid_table:
            pid_table[pid] = i
            i += 1
        pid = pid_table[pid]

        basename = os.path.basename(fname)
        record = record.astype(np.float32)

        start = np.random.randint(length - (args.sec * sample_rate))

        data = {}
        data['patient_id'] = pid
        data['curr_sample_rate'] = sample_rate
        data['feats'] = record[start: start + (args.sec * sample_rate)]
        scipy.io.savemat(os.path.join(dest_path, f"{pid}_{basename}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)