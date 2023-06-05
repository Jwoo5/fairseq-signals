import os
import argparse

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
    meta_path = os.path.realpath(args.meta_dir)

    if args.rebase:
        import shutil
        shutil.rmtree(dest_path)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    try:
        csv = pd.read_csv(os.path.join(meta_path, 'ptbxl_database_translated.csv'))
    except FileNotFoundError:
        raise FileNotFoundError(
            "cannot find the metadata file (ptbxl_database_translated.csv) "
            "please ensure that this file is located in --meta-dir "
            "or download from ..."
            f"--meta-dir: {meta_path}"
        )

    exclude = None
    if args.exclude is not None:
        exclude = []
        with open(args.exclude, "r") as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                exclude.append(items[1])

    fnames = csv['filename_hr'].to_numpy()
    reports = csv['report_en'].to_numpy()
    scp_codes = csv['scp_codes']

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