"""Record extraction code example and guide."""

import os
import glob
import argparse

import pandas as pd

from fairseq_signals.utils.file import remove_common_segments

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
        help="Path to the raw data directory.",
    )

    return parser

def main(args):
    os.makedirs(args.processed_root, exist_ok=True)

    # TODO - Collect files or load from existing records file
    files = ...

    files = pd.Series(
        files,
        name='path',
    )

    # Remove path segments common to all files such that the raw dataset directory can
    # move and the record paths can be easily updated using a path join
    files = remove_common_segments(files)
    records = files.to_frame()

    # Optionally define dataset column if more than one dataset in this source and said
    # information is conveniently available
    records['dataset'] = ...

    # Save records
    records.to_csv(os.path.join(args.processed_root, 'records.csv'), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
