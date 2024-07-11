"""Record extraction code specific to PhysioNet 2021 v1.0.3."""

import os
import glob
import argparse

import pandas as pd

from fairseq_signals.utils.file import remove_common_segments, remove_ext

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="Directory to save the processed data.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Path to the PhysioNet 2021 directory.",
    )

    return parser

def main(args):
    os.makedirs(args.processed_root, exist_ok=True)

    if os.path.normpath(args.raw_root).split(os.sep)[-1] != 'training':
        args.raw_root = os.path.join(args.raw_root, 'training')

    # Collect files
    files = remove_ext(pd.Series(
        glob.iglob(os.path.join(args.raw_root, "**/**/*.mat")),
        name='path',
    ))

    # Make sure that the corresponding .hea file exists (certain ones don't in ningbo)
    files = files[(files + '.hea').apply(os.path.isfile)]

    files = remove_common_segments(files)
    records = files.to_frame()

    # Extract dataset name
    records['dataset'] = records['path'].str.extract(r'^[\\/]?([^\\/]+)', expand=False)

    # Save records
    records.to_csv(os.path.join(args.processed_root, 'records.csv'), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
