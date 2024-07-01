from typing import Union
import os
import argparse

import pandas as pd

from fairseq_signals.utils.splits import DatasetSplitter

def bool_or_str(value):
    if value in ['true', 'True', 't', 'T']:
        return True
    elif value in ['false', 'False', 'f', 'F']:
        return False

    return value

def get_parser():
    parser = argparse.ArgumentParser(description="Process and split the dataset.")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Split strategy in 'random', 'grouped', 'temporal', and 'grouped_temporal'."
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="Root directory for processed data."
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="meta.csv",
        help="Filename for metadata."
    )
    parser.add_argument(
        "--segmented_file",
        type=str,
        default="segmented.csv",
        help="Filename for segmented data (optional)."
    )
    parser.add_argument(
        "--save_files_suffix",
        type=str,
        default="_split",
        help="Suffix added to saved split files. Do not set as '' or it will overwrite "
            "the existing files."
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default='0.8,0.1,0.1',
        help="Comma-delimited decimal percentages for splits, e.g., '0.8,0.1,0.1'."
    )
    parser.add_argument(
        "--split_labels",
        type=str,
        default='train,valid,test',
        help="Comma-delimited names for the dataset splits, e.g., 'train,valid,test'."
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default=None,
        help="Column defining groups. Must specify when using a grouped strategy."
    )
    parser.add_argument(
        "--date_col",
        type=str,
        default=None,
        help="Column defining groups. Must specify when using a grouped strategy."
    )
    parser.add_argument(
        "--grouped_temporal_filter_strategy",
        type=bool_or_str,
        default="train",
        help="Grouped temporal filtering strategy. If False, there is no filtering based on "
            "temporal versus group assignments. If True, filters overlap across all "
            "splits. Otherwise, expects a split label with which to filter overlap "
            "between it and all other splits. E.g., using filter_strategy='train' "
            "could be used to filter overlap with evaluative splits."
    )
    parser.add_argument(
        "--filter_cols",
        type=str,
        default=None,
        help="Boolean columns in meta to filter out samples (excluded if any True)."
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="Subset of datasets to include."
    )
    parser.add_argument(
        "--keep_cols",
        type=str,
        default=None,
        help="Columns from the metadata to keep in the new files."
    )

    return parser

def process_str_list(text_lst):
    return pd.Series(text_lst).str.split(',').explode().str.strip()

def main(args):
    # Load the metadata
    meta = pd.read_csv(os.path.join(args.processed_root, args.meta_file))

    # Subset datasets
    if args.dataset_subset is not None:
        dataset_subset = pd.Series(
            args.dataset_subset
        ).str.split(',').explode().str.strip().values
        meta = meta[meta['dataset'].isin(dataset_subset)].copy()

    # Remove samples based on filter columns, e.g., those with NaNs or missing leads
    if args.filter_cols is not None:
        filter_cols = pd.Series(args.filter_cols).str.split(',').explode().str.strip().values
        meta = meta[~meta[filter_cols].any(axis=1)].copy()

    # Split the data
    kwargs = {}
    if args.strategy == 'grouped_temporal':
        kwargs['filter_strategy'] = args.grouped_temporal_filter_strategy

    splitter = DatasetSplitter(
        args.strategy,
        args.fractions,
        group_col=args.group_col,
        date_col=args.date_col,
        split_labels=process_str_list(args.split_labels).values,
        split_col='split',
        **kwargs
    )
    meta = splitter(meta)

    # Determine which columns to keep in the files being saved
    keep_cols = ['idx', 'save_file', 'split']
    if args.group_col is not None:
        keep_cols.append(args.group_col)

    if args.strategy == 'grouped_temporal':
        keep_cols.append(f'{args.group_col}_split')

    if args.strategy == 'temporal' or args.strategy == 'grouped_temporal':
        keep_cols.append(args.date_col)

    if args.keep_cols is not None:
        keep_cols.extend(process_str_list(args.keep_cols).values)

    meta = meta[keep_cols].copy()

    meta.to_csv(
        os.path.join(args.processed_root, f'meta{args.save_files_suffix}.csv'),
        index=False,
    )

    # Also split the segmented data if provided
    if args.segmented_file is not None:
        segmented = pd.read_csv(os.path.join(args.processed_root, args.segmented_file))
        segmented = segmented.merge(
            meta,
            on='save_file',
            how='left',
        )

        # Remove segments corresponding to those samples removed from meta
        segmented = segmented.dropna(subset=['idx'])

        segmented[keep_cols + ['path', 'sample_size']].to_csv(
            os.path.join(args.processed_root, f'segmented{args.save_files_suffix}.csv'),
            index=False,
        )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
