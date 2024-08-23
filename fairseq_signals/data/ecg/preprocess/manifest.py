import argparse
import glob
import os
import random
from tqdm import tqdm

import scipy.io

"""
    Usage: python path/to/manifest.py \
            /path/to/data \
            --dest /path/to/manifest \
            --ext $ext \
            --valid-percent $valid
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing mat files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        type=str,
        help="group the filenames using this as a delimiter so that the same group should go into "
            "the same split. when it is given, the filenames are grouped by "
            "the basename before the last `group_by` token. e.g., with `group_by='_'`, 'a_1.mat' "
            "and 'a_2.mat' should go to the same split."
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 0.5

    root_path = os.path.realpath(args.root)
    groupby = args.group_by
    rand = random.Random(args.seed)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "valid.tsv"), "w") as valid_f, open(
        os.path.join(args.dest, "test.tsv"), "w"
    ) as test_f:
        print(root_path, file=train_f)
        print(root_path, file=valid_f)
        print(root_path, file=test_f)

        def write(fnames, dest, split=None):
            for fname in tqdm(fnames, total=len(fnames), desc=split):
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue

                if args.ext == 'mat':
                    data = scipy.io.loadmat(file_path)
                    ecg_length = data['feats'].shape[-1]

                    print(
                        "{}".format(os.path.relpath(file_path, root_path)), file=dest, end='\t'
                    )
                    print(ecg_length, file=dest)
                else:
                    raise AssertionError(args.ext)

        search_path = os.path.join(args.root, "**/*." + args.ext)
        fnames = glob.glob(search_path, recursive=True)
        if groupby is not None:
            grouped_fnames = dict()
            for fname in fnames:
                group = fname[:fname.rindex(groupby)]
                if group not in grouped_fnames:
                    grouped_fnames[group] = []
                grouped_fnames[group].append(fname)
            
            groups = list(grouped_fnames.keys())
            rand.shuffle(groups)

            valid_len = int(len(groups) * args.valid_percent)
            test_len = int(len(groups) * args.valid_percent)
            train_len = len(groups) - (valid_len + test_len)

            train = sum([grouped_fnames[x] for x in groups[:train_len]], [])
            valid = sum([grouped_fnames[x] for x in groups[train_len:train_len + valid_len]], [])
            test = sum([grouped_fnames[x] for x in groups[train_len + valid_len:]], [])
            
            write(train, train_f, split="train")
            write(valid, valid_f, split="valid")
            write(test, test_f, split="test")
        else:
            rand.shuffle(fnames)

            valid_len = int(len(fnames) * args.valid_percent)
            test_len = int(len(fnames) * args.valid_percent)
            train_len = len(fnames) - (valid_len + test_len)

            train = fnames[:train_len]
            valid = fnames[train_len:train_len + valid_len]
            test = fnames[train_len + valid_len:]

            write(train, train_f, split="train")
            write(valid, valid_f, split="valid")
            write(test, test_f, split="test")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)