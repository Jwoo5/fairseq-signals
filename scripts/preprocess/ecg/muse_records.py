"""Record extraction code specific to extracting MUSE v8.0.2.10132 ECGs."""

import os
import glob
import argparse

import pandas as pd

from fairseq_signals.utils.file import remove_common_segments

SOURCE = 'uhn'

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
        help="Path to the directory with (subdirectories of) MUSE .xml files.",
    )

    return parser

def main(args):
    os.makedirs(args.processed_root, exist_ok=True)

    # Collect files
    files = remove_ext(pd.Series(
        glob.glob(os.path.join(args.raw_root, f"/**/*.xml"), recursive=True),
        name='path',
    ))

    files = remove_common_segments(files)
    records = files.to_frame()

    # Save records
    records.to_csv(os.path.join(args.processed_root, 'records.csv'), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
