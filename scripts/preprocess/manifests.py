import os
import argparse

import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description="Process and split the dataset.")
    parser.add_argument(
        "--split_file_paths",
        type=str,
        required=True,
        help="Comma-delimited split file paths."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Save directory for manifests."
    )
    
    return parser

def process_str_list(text_lst):
    return pd.Series(text_lst).str.split(',').explode().str.strip()

def main(args):
    # Load manifests
    split_manifests = process_str_list(args.split_file_paths).apply(pd.read_csv).values
    splits = pd.concat(split_manifests)

    # Determine the path segments in common
    common_prefix = os.path.commonprefix(list(splits['path'].values))
    if common_prefix[-1] != "/":
        common_prefix = common_prefix[:common_prefix.rfind("/") + 1]

    splits['remaining'] = splits['path'].str.slice(start=len(common_prefix))

    # Create and save the manifests
    for split, group in splits.groupby('split'):
        manifest = pd.DataFrame(
            {common_prefix: group['sample_size']}
        ).set_index(group['remaining'])
        manifest.index.name = None
        manifest.to_csv(os.path.join(args.save_dir, f'{split}.tsv'), sep='\t')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
