"""Label preparation code. Tested with PhysioNet 2021 v1.0.3.

The weights and weight abbreviation files can be obtained here:
https://github.com/physionetchallenges/evaluation-2021
"""
import os
import argparse

import logging

import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="Path to save the processed data root directory.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Path to save resultant labels file. Defaults to "
            "'{processed_root}/labels/labels.csv'.",
    )

    return parser

def get_labels(meta_file):
    # Get SNOMED-CT diagnoses
    meta = pd.read_csv(meta_file, index_col='idx').sort_index()
    labels = meta[[
        'is_male',
        '1dAVb',
        'RBBB',
        'LBBB',
        'SB',
        'ST',
        'AF',
        'normal_ecg',
    ]].copy()

    return labels

def main(args):
    labels = get_labels(os.path.join(args.processed_root, 'meta.csv'))

    if args.labels_path is None:
        os.makedirs(os.path.join(args.processed_root, 'labels'), exist_ok=True)
        labels_path = os.path.join(args.processed_root, 'labels', 'labels.csv')
    else:
        labels_path = args.labels_path

    labels.to_csv(labels_path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
