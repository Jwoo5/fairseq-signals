import os
import argparse

import pandas as pd
import numpy as np
import wfdb
import scipy.io
from tqdm import tqdm

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
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    database = pd.read_csv(os.path.join(dir_path, "machine_measurements.csv"))
    record_list = pd.read_csv(os.path.join(dir_path, "record_list.csv")).set_index("study_id")

    exclude = None
    if args.exclude is not None:
        exclude = []
        with open(args.exclude, "r") as f:
            for line in f.readlines():
                items = line.strip().split("\t")
                exclude.append(items[1])
        exclude = set(exclude)
    
    study_ids = database["study_id"].to_numpy()
    reports = []
    n_reports = 18
    for i, row in tqdm(database.iterrows(), total=len(database)):
        report_txt = ""
        for j in range(n_reports):
            report = row[f"report_{j}"]
            if type(report) == str:
                report_txt += report + " "
        report_txt = report_txt[:-1]
        reports.append(report_txt)
    
    n = 0
    for study_id, report in tqdm(zip(study_ids, reports), total=len(reports)):
        fname = record_list.loc[study_id]["path"]

        if exclude is not None and study_id in exclude:
            n += 1
            continue

        basename = os.path.basename(fname)
        record = wfdb.rdsamp(os.path.join(dir_path, fname))
        sample_rate = record[1]["fs"]
        record = record[0].T

        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        data = {}
        data['curr_sample_rate'] = sample_rate
        data['feats'] = record
        data['text'] = report
        scipy.io.savemat(os.path.join(dest_path, basename + ".mat"), data)

    # print('excluded: ' + str(n))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)