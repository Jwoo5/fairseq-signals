import argparse
import glob
import os
import random
import warnings

import scipy.io

"""
    Usage: python path/to/manifest_physionet2021.py \
            /path/to/signals \
            --pretrain_subset $pretrain_subsets \
            --finetune_subset $finetune_subsets \
            --dest /path/to/manifest \
            --ext $ext \
            --pretrain-valid-percent $valid-pretrain
            --finetune-valid-percent $valid-finetune
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing mat files to index"
    )
    parser.add_argument(
        "--pretrain_subset",
        default="cpsc_2018, cpsc_2018_extra, georgia, ptb-xl, chapman_shaoxing, ningbo",
        type=str,
        help="comma seperated list of data subsets to manifest for pre-training (e.g. cpsc_2018, cpsc_2018_extra, ...), "
             "each of which should be corresponded with the name of the sub-directory"
    )
    parser.add_argument(
        "--finetune_subset",
        default="cpsc_2018, georgia",
        type=str,
        help="comma seperated list of data subsets for fine-tuning (e.g. cpsc_2018, cpsc_2018_extra, ...), "
             "each of which should be corresopnded with the name of the sub-directory. Note that "
             "the training set of these datasets will be also used for pre-training, but the "
             "validation and test sets will not be included in the pre-training dataset."
    )

    parser.add_argument(
        "--pretrain-valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set for pre-training (between 0 and "
             "0.5)",
    )
    parser.add_argument(
        "--finetune-valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set for fine-tuning (between 0 and "
             "0.5). these splits will not be included in the pre-training dataset",
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
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
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
    assert args.pretrain_valid_percent >= 0 and args.pretrain_valid_percent <= 0.5
    assert args.finetune_valid_percent >= 0 and args.finetune_valid_percent <= 0.5

    root_path = os.path.realpath(args.root)
    pretrain_subset = [x.strip() for x in args.pretrain_subset.split(",")]
    finetune_subset = [x.strip() for x in args.finetune_subset.split(",")]
    groupby = args.group_by
    rand = random.Random(args.seed)

    def random_split(fnames, valid_percent):
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
        else:
            rand.shuffle(fnames)

            valid_len = int(len(fnames) * valid_percent)
            test_len = int(len(fnames) * valid_percent)
            train_len = len(fnames) - (valid_len + test_len)

            train = fnames[:train_len]
            valid = fnames[train_len:train_len + valid_len]
            test = fnames[train_len + valid_len:]

        return train, valid, test

    if not os.path.exists(os.path.join(args.dest, "pretrain")):
        os.makedirs(os.path.join(args.dest, "pretrain"))
    if not os.path.exists(os.path.join(args.dest, "finetune")):
        os.makedirs(os.path.join(args.dest, "finetune"))

    with open(os.path.join(args.dest, "pretrain/train.tsv"), "w") as pretrain_f, open(
        os.path.join(args.dest, "pretrain/valid.tsv"), "w") as pre_valid_f, open(
        os.path.join(args.dest, "pretrain/test.tsv"), "w") as pre_test_f, open(
        os.path.join(args.dest, "finetune/train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "finetune/valid.tsv"), "w") as valid_f, open(
        os.path.join(args.dest, "finetune/test.tsv"), "w"
    ) as test_f:
        print(root_path, file=pretrain_f)
        print(root_path, file=train_f)
        print(root_path, file=valid_f)
        print(root_path, file=test_f)

        def write(fnames, dest):
            for fname in fnames:
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue

                if args.ext == "mat":
                    data = scipy.io.loadmat(file_path)
                    if "feats" not in data:
                        raise AssertionError(
                            "each data file should contain ECG signals as a value of the key "
                            "'feats' for support efficient batching in the training step."
                        )
                    length = data["feats"].shape[-1]

                    print(
                        "{}".format(os.path.relpath(file_path, root_path)), file=dest, end='\t'
                    )
                    print(length, file=dest)
                else:
                    raise AssertionError(f"extension for {args.ext} is not allowed.")

        already_processed = []

        for s in finetune_subset:
            search_path = os.path.join(args.root, s, "**/*." + args.ext)
            fnames = glob.glob(search_path, recursive=True)
            if len(fnames) == 0:
                warnings.warn(
                    f"No files found in {os.path.join(args.root, s)} directory. Please make sure "
                    f"{os.path.join(args.root, s)} contains {args.ext} files to be processed."
                )

            train, valid, test = random_split(fnames, args.finetune_valid_percent)

            if s in pretrain_subset:
                already_processed.append(s)
                # need to re-split only the train subset for pre-training dataset
                pretrain, pre_valid, pre_test = random_split(train, args.pretrain_valid_percent)
                write(pretrain, pretrain_f)
                write(pre_valid, pre_valid_f)
                write(pre_test, pre_test_f)

            write(train, train_f)
            write(valid, valid_f)
            write(test, test_f)

        for s in pretrain_subset:
            if s in already_processed:
                continue

            search_path = os.path.join(args.root, s, "**/*." + args.ext)
            fnames = glob.glob(search_path, recursive=True)
            if len(fnames) == 0:
                warnings.warn(
                    f"No files found in {os.path.join(args.root, s)} directory. Please make sure"
                    f"{os.path.join(args.root, s)} contains {args.ext} files to be processed."
                )
            
            train, valid, test = random_split(fnames, args.pretrain_valid_percent)

            write(train, pretrain_f)
            write(valid, pre_valid_f)
            write(test, pre_test_f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)