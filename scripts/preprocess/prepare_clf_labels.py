import os
import argparse

import yaml

import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to output directory.',
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help="Path to a non-subsetted labels CSV containing an 'idx' column.",
    )
    parser.add_argument(
        '--meta_splits',
        type=str,
        required=True,
        help="Path to meta splits CSV containing 'idx' and 'split' columns. ",
    )
    parser.add_argument(
        '--segmented_splits',
        type=str,
        required=False,
        help="Path to segmented splits CSV containing 'idx' and 'split' columns. ",
    )
    parser.add_argument(
        '--labeled_meta_splits_save_file',
        required=False,
        type=str,
        help='Path to save the meta splits CSV, as filtered to those '
            'samples having labels. Useful for creating task-specific manifests.',
    )
    parser.add_argument(
        '--labeled_segmented_splits_save_file',
        required=False,
        type=str,
        help='Path to save the segmented splits CSV, as filtered to those '
            'samples having labels. Useful for creating task-specific manifests.',
    )

    return parser

def main(args):
    if args.labeled_segmented_splits_save_file is not None:
        if args.segmented_splits is None:
            raise ValueError(
                'Must specify `segmented_splits` when '
                '`labeled_segmented_splits_save_file` is specified.'
            )

    labels = pd.read_csv(args.labels, index_col='idx').sort_index()
    label_names = list(labels.columns)

    if not labels.index.is_unique:
        raise ValueError('Expected unique sample IDs (idx) in labels.')

    meta_splits = pd.read_csv(args.meta_splits, index_col='idx')

    # Create label array
    y = np.zeros((max([labels.index.max(), meta_splits.index.max()]) + 1, len(label_names)))
    np.put_along_axis(y, labels.index.values.reshape(-1, 1), labels.values, 0)

    # When calculating label metadata, use only those labeled samples in the splits
    meta_splits = meta_splits.join(labels, how='left')
    label_missing = meta_splits[label_names].isna().any(axis=1)
    if label_missing.any():
        print(
            f'Missing {label_missing.sum()} labels. Filling unlabelled samples in '
            'y.npy with zeros. Ensure that these samples do not appear in the '
            'manifest by specifying and using the labeled save files.'
        )

    meta_splits = meta_splits[~label_missing].copy()

    # Create label definition
    label_counts = meta_splits[label_names + ['split']].groupby('split').sum()
    label_percents = label_counts.div(meta_splits['split'].value_counts(), axis=0)
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

    label_def['pos_count_all'] = meta_splits[label_names].sum(axis=0)
    label_def['pos_percent_all'] = label_def['pos_count_all'] / len(meta_splits)

    # Save the label definition
    label_def.to_csv(os.path.join(args.output_dir, 'label_def.csv'))

    # Save the label array
    np.save(os.path.join(args.output_dir, 'y.npy'), y)

    with open(os.path.join(args.output_dir, 'prepare_clf_labels_args.yaml'), 'w') as file:
        yaml.dump(args, file)

    if 'pos_weight_train' in label_def:
        # Save pos_weight as a string (easy to load into terminal)
        pos_weight_str = '[' + ','.join(
            (label_def['pos_weight_train']).round(decimals=6).values.astype('str')
        ) + ']'
        with open(os.path.join(args.output_dir, 'pos_weight.txt'), 'w') as f:
            f.write(pos_weight_str)

    if args.labeled_meta_splits_save_file is not None:
        meta_splits.drop(label_names, axis=1, inplace=True)
        meta_splits.to_csv(args.labeled_meta_splits_save_file)

    if args.labeled_segmented_splits_save_file is not None:
        segmented_splits = pd.read_csv(args.segmented_splits, index_col='idx')
        segmented_splits = segmented_splits[
            segmented_splits.index.isin(meta_splits.index)
        ].copy()
        segmented_splits.to_csv(args.labeled_segmented_splits_save_file)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
