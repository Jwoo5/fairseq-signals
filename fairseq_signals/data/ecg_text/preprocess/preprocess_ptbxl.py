import os
import argparse
import warnings

import pandas as pd
import numpy as np
import wfdb
import scipy.io

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', metavar='DIR', default='.',
        help='root directory containing ptbxl files to pre-process'
    )
    parser.add_argument(
        '--dest', type=str, metavar='DIR', default='.',
        help='output directory'
    )
    parser.add_argument(
        '--rebase', action='store_true',
        help='if set, remove and create directory for --dest'
    )
    parser.add_argument(
        '--exclude', type=str, default=None,
        help='path to .tsv file consisting of (index, ecg_id) to be excluded'
    )

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    if args.rebase:
        import shutil
        shutil.rmtree(dest_path)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if not os.path.exists(os.path.join(dir_path, "ptbxl_database_translated.csv")):
        warnings.warn(
            "Cannot find ptbxl_database_translated.csv which is an English-translated version of "
            "the original database file. We instead use ptbxl_database.csv where some reports "
            "are written in Deutsch, which may affect the training. "
            "Please download https://github.com/Jwoo5/fairseq-signals/blob/master/fairseq_signals/data/ecg_text/preprocess/ptbxl_database_translated.csv "
            "and ensure that this file is located in the root directory for the better performance."
        )
        database = pd.read_csv(os.path.join(dir_path, "ptbxl_database.csv"))
    else:
        database = pd.read_csv(os.path.join(dir_path, "ptbxl_database_translated.csv"))

    exclude = None
    if args.exclude is not None:
        exclude = []
        with open(args.exclude, "r") as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                exclude.append(items[1])
        exclude = set(exclude)

    fnames = database['filename_hr'].to_numpy()
    reports = database['report'].to_numpy()
    scp_codes = database['scp_codes']

    n = 0
    for fname, report, scp_code in zip(fnames, reports, scp_codes):
        ecg_id = str(int(os.path.basename(fname).split('_')[0]))
        if exclude is not None and ecg_id in exclude:
            n += 1
            continue

        basename = os.path.basename(fname)
        record = wfdb.rdsamp(os.path.join(dir_path, fname))
        sample_rate = record[1]['fs']
        record = record[0].T

        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        data = {}
        data['curr_sample_rate'] = sample_rate
        data['feats'] = record
        data['text'] = report
        data['diagnoses'] = list(eval(scp_code).keys())
        scipy.io.savemat(os.path.join(dest_path, basename + '.mat'), data)

    # print('excluded: ' + str(n))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)