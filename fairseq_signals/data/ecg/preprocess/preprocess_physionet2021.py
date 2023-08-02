"""
Data pre-processing: encode labels (age, diagnosis, patient id) and crop data.
"""

import argparse
import os
import functools
import math
import linecache
import glob
import wfdb
import numpy as np
from scipy.interpolate import interp1d
import scipy.io

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
        default="cpsc_2018, cpsc_2018_extra, georgia, ptb-xl, chapman_shaoxing, ningbo",
        type=str,
        help="comma separated list of sub-directories of data subsets to be preprocessed, "
        "each of which is labeled seperately (e.g. cpsc_2018, cpsc_2018_extra, ...)"
    )
    parser.add_argument(
        "--leads",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        type=str,
        help="comma separated list of lead numbers. (e.g. 0,1 loads only lead I and lead II)"
        "note that the order is following: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
    )
    parser.add_argument(
        "--sample-rate",
        default=500,
        type=int,
        help="if set, data must be sampled by this sampling rate to be processed"
    )
    parser.add_argument(
        "--resample",
        default=False,
        action='store_true',
        help='if set, resample data to have a sample rate of --sample-rate'
    )
    parser.add_argument("--dest", type=str, metavar="DIR",
                       help="output directory")
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

    try:
        classes, _ = ecg_utils.get_physionet_weights(meta_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "cannot find the metadata file for labeling (weights.csv). "
            "Please ensure that the file is located in --meta-dir. "
            "You can download it from https://github.com/physionetchallenges/evaluation-2021. "
            f"--meta-dir: {meta_path}"
        )

    leads = args.leads.replace(' ','').split(',')
    leads_to_load = [int(lead) for lead in leads]
    subset = args.subset.replace(' ','').split(',')

    if len(glob.glob(os.path.join(args.root, "**/**/*." + args.ext))) == 0:
        args.root = os.path.join(args.root, "training")

    pid_table = dict()
    for i, fname in enumerate(glob.iglob(os.path.join(args.root, "**/**/*." + args.ext))):
        pid_table[os.path.basename(fname)[:-4]] = i

    for s in subset:
        if not os.path.exists(os.path.join(args.dest, s)):
            os.makedirs(os.path.join(args.dest, s))

        dir_path = os.path.realpath(os.path.join(args.root, s))
        search_path = os.path.join(dir_path, "**/*." + args.ext)

        fnames = list(glob.iglob(search_path, recursive=True))
        chunk_size = math.ceil(len(fnames) / args.workers)

        file_chunks = [fnames[i:i+chunk_size] for i in range(0, len(fnames), chunk_size)]

        func = functools.partial(
            preprocess,
            args,
            pid_table,
            classes,
            os.path.join(args.dest, s),
            leads_to_load
        )
        pool = Pool(processes = args.workers)
        pool.map(func, file_chunks)
        pool.close()
        pool.join()

def preprocess(args, pid_table, classes, dest_path, leads_to_load, fnames):
    for fname in fnames:
        fname = fname[:-(len(args.ext)+1)]

        if not os.path.exists(fname + ".hea"):
            continue

        y = set(linecache.getline(fname + ".hea", 16).replace(',',' ').split()[1:])
        label = np.zeros(len(classes), dtype=bool)
        for i, x in enumerate(classes):
            if x & y:
                label[i] = 1

        age = linecache.getline(fname + ".hea", 14).split()
        assert len(age) == 3, fname
        try:
            age = int(age[-1])
        except ValueError:
            age = 0
        
        sex = linecache.getline(fname + ".hea", 15).split()
        assert len(sex) == 3, fname
        sex = 0 if sex[-1] == "Male" else 1

        try:
            age = int(linecache.getline(fname + '.hea', 14).split()[1])
        except ValueError:
            age = 0
        sex = 0 if linecache.getline(fname + '.hea', 15).split()[1] == "Male" else 1

        sample_rate = int(linecache.getline(fname + '.hea', 1).split()[2])

        if (
            args.sample_rate
            and sample_rate != args.sample_rate
            and not args.resample
        ):
            continue

        annot = wfdb.rdheader(
            os.path.splitext(fname)[0]
        ).__dict__

        sample = scipy.io.loadmat(fname)['val']
        adc_gains = np.array(annot['adc_gain'])[:, None]

        if np.isnan(sample).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        sample = sample / adc_gains

        length = sample.shape[-1]

        if (
            args.sample_rate
            and sample_rate != args.sample_rate
            and args.resample
        ):
            sample_size = length * (args.sample_rate / sample_rate)
            x = np.linspace(0, sample_size - 1, length)
            f = interp1d(x, sample, kind='linear')
            sample = f(list(range(int(sample_size))))
            sample_rate = args.sample_rate
            length = int(sample_size)

        pid = pid_table[os.path.basename(fname)]
        for i, seg in enumerate(range(0, length, int(args.sec * sample_rate))):
            data = {}
            data['age'] = age
            data['sex'] = sex
            data['label'] = label
            data['patient_id'] = pid
            data['curr_sample_rate'] = sample_rate
            if seg + args.sec * sample_rate <= length:
                data['feats'] = sample[leads_to_load, seg: int(seg + args.sec * sample_rate)]
                scipy.io.savemat(os.path.join(dest_path, os.path.basename(fname) + f"_{i}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)