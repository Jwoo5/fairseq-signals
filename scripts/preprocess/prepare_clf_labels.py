import os
import argparse

import yaml

import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to a non-subsetted labels file which contains an 'idx' column.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Path to splits file which contains 'idx' and 'split' columns. "
            "The sample IDs in the labels must be a superset of those in the splits.",
    )

    return parser

def main(args):
    labels = pd.read_csv(args.labels, index_col='idx').sort_index()
    label_names = list(labels.columns)

    if not (labels.index == np.arange(labels.index.min(), labels.index.max() + 1)).all():
        raise ValueError('Expected contiguous sample IDs (idx).')

    if not labels.index.is_unique:
        raise ValueError('Expected unique sample IDs (idx).')

    splits = pd.read_csv(args.splits, index_col='idx')

    # Get those labels being used in the splits
    splits = splits.join(labels, how='left')

    # Create label definition
    label_counts = splits[label_names + ['split']].groupby('split').sum()
    label_percents = label_counts.div(splits['split'].value_counts(), axis=0)
    label_counts = label_counts.T.add_prefix('pos_count_')

    label_weights = (1 - label_percents)/label_percents
    label_percents = label_percents.T.add_prefix('pos_percent_')
    label_weights = label_weights.T.add_prefix('pos_weight_')
    label_def = pd.concat([
        label_counts,
        label_percents,
        label_weights,
    ], axis=1)
    label_def.index.name = 'name'

    label_def['pos_count_all'] = splits[label_names].sum(axis=0)
    label_def['pos_percent_all'] = label_def['pos_count_all'] / len(splits)

    # Extract and pad labels
    y = labels.values.astype(float)
    y_empty = np.zeros((labels.index.min(), y.shape[1]), dtype=y.dtype)
    y = np.concatenate([y_empty, y])

    # Save the label definition
    label_def.to_csv(os.path.join(args.output_dir, 'label_def.csv'))

    # Save the label array
    np.save(os.path.join(args.output_dir, 'y.npy'), y)

    # Save pos_weight as a string (easy to load into terminal)
    pos_weight_str = '[' + ','.join(
        (label_def['pos_weight_train']).round(decimals=6).values.astype('str')
    ) + ']'
    with open(os.path.join(args.output_dir, 'pos_weight.txt'), 'w') as f:
        f.write(pos_weight_str)

    with open(os.path.join(args.output_dir, 'prepare_clf_labels_args.yaml'), 'w') as file:
        yaml.dump(args, file)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
