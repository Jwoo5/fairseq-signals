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
        "--weights_path",
        type=str,
        required=True,
        help="Path to weights file - see "
            "https://github.com/physionetchallenges/evaluation-2021",
    )
    parser.add_argument(
        "--weight_abbrev_path",
        type=str,
        required=True,
        help="Path to weight abbreviations file - see "
            "https://github.com/physionetchallenges/evaluation-2021",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Path to save resultant labels file. Defaults to "
            "'{processed_root}/labels/labels.csv'.",
    )
    parser.add_argument(
        "--uncombined",
        action="store_true", 
        help="Do not combine the labels as specified in the weights file.",
    )

    return parser

def get_labels(meta_file, weights_path, weight_abbrev_path, combined=True):
    # Get SNOMED-CT diagnoses
    meta = pd.read_csv(meta_file, index_col='idx').sort_index()
    diagnoses = meta["diagnosis"].str.split(",").explode()

    # Get combined weights
    weights = pd.read_csv(
        weights_path,
        index_col="Unnamed: 0",
    ).index.to_series().str.split("|")
    weights = weights.rename("diagnosis").to_frame()
    weights["abbrev"] = pd.read_csv(
        weight_abbrev_path,
        index_col="Unnamed: 0",
    ).index
    weights.reset_index(drop=True, inplace=True)

    # Compute uncombined weights
    weights_expl = weights["diagnosis"].explode()

    # Drop if diagnosis unused
    use = diagnoses.isin(weights_expl)
    logging.info(
        f"Using {use.sum() / len(diagnoses)*100:.2f}% of available diagnoses."
    )
    diagnoses = diagnoses[use]

    if args.uncombined:
        weights_expl.reset_index(drop=True, inplace=True)
        labels = weights['abbrev'].str.split('|').explode()
    else:
        labels = weights['abbrev']

    # Map diagnoses to label indices
    ind_map = dict(zip(weights_expl.values, weights_expl.index))
    diagnoses = diagnoses.map(ind_map)

    # Create label DataFrame
    y = np.zeros((meta.index.max() + 1, len(labels)))
    y[diagnoses.index.values, diagnoses.values] = 1

    labels = pd.DataFrame(y, columns=labels)
    labels = labels[meta.index.min():].set_index(meta.index).astype(int)

    return labels

def main(args):
    labels = get_labels(
        os.path.join(args.processed_root, 'meta.csv'),
        args.weights_path,
        args.weight_abbrev_path,
        combined=(not args.uncombined),
    )

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
