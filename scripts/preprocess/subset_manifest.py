"""Subset train manifest to a specified percentage. Useful for data scaling experiments.

Copies any other .tsv files (presumably evaluative manifests), keeping them the same.

Assumes all .tsv files in the `manifest_dir` are manifest files. Must have 'train.tsv'.
"""
import os
import shutil
import re

import argparse

import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(
        description='Subset train manifest to a specified percentage.'
    )
    parser.add_argument(
        '--manifest_dir',
        required=True,
        type=str,
        help='Directory containing the original manifests',
    )
    parser.add_argument(
        '--percent',
        required=True,
        type=float,
        help='Decimal percentage of samples to keep',
    )
    parser.add_argument(
        '--save_dir',
        required=False,
        type=str,
        help='Directory to save the subset manifest. Defaults to {manifest_dir}/{percent}.',
    )

    return parser

def main(manifest_dir, percent, save_dir=None):
    """
    Create a subset of the training manifest and copy other .tsv files.

    This function loads the train.tsv file, creates a subset based on the specified
    percentage, saves the subset, and copies other .tsv files to the save directory.

    Parameters
    ----------
    manifest_dir : str
        Directory containing the original manifests.
    percent : float
        Percentage of the original dataset to use (as a decimal).
    save_dir : str, optional
        Directory to save the subset manifest. If not provided, a new directory
        is created within manifest_dir.

    Raises
    ------
    ValueError
        If the save directory already exists.
    """
    # Load the train manifest
    train = pd.read_csv(os.path.join(manifest_dir, 'train.tsv'), sep='\t')

    # Remove segment number to create a 'save_file' column using regex
    train['save_file'] = \
        train['Unnamed: 0'].str.replace(r'_\d+\.mat$', '.mat', regex=True)

    # Create a subset based on the specified percentage
    unique_save_files = pd.Series(train['save_file'].unique())
    subset_save_files = unique_save_files.sample(frac=percent)
    train_subset = train[train['save_file'].isin(subset_save_files)]

    # Remove the 'save_file' column before saving
    train_subset.drop('save_file', axis=1, inplace=True)

    # Create save directory if not specified
    if save_dir is None:
        save_dir = os.path.join(manifest_dir, str(percent))

    # Check if save directory already exists
    if os.path.exists(save_dir):
        raise ValueError(f"Save directory {save_dir} already exists.")

    # Create save directory
    os.makedirs(save_dir)

    # Save the subset
    train_subset.set_index('Unnamed: 0', inplace=True)
    train_subset.index.name = "" # Label as empty string rather than 'Unnamed: 0'
    print(train_subset)
    print(train_subset.columns)
    train_subset.to_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', index=True)

    # Copy other .tsv files
    for file in os.listdir(manifest_dir):
        if file.endswith('.tsv') and file != 'train.tsv':
            shutil.copy2(os.path.join(manifest_dir, file), os.path.join(save_dir, file))

    print(f"Train subset and other .tsv files saved to {save_dir}")
    print(f"Train subset size: {len(train_subset)} / {len(train)} = {len(train_subset)/len(train):.6f}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.manifest_dir, args.percent, args.save_dir)