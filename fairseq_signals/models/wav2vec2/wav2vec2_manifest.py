import argparse
import glob
import os
import random

"""
    Usage: python path/to/wav2vec2_manifest.py \
            /path/to/signals \
            --subset $subsets
            --dest /manifest/path \
            --predir /sub/root/dir \
            --ext $ext \
            --valid-percent $valid
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--subset",
        default="CPSC2018, CPSC2018_2, Ga, PTBXL, ChapmanShaoxing, Ningbo",
        type=str,
        help="comma seperated list of data subsets to manifest (e.g. CPSC2018, CPSC2018_2, ...), "
             "each of which should be a name of the sub-directory"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--predir", default=".", type=str, metavar="DIR", help="if set, create sub-root directory in --dest"
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
    subset = args.subset.replace(' ', '').split(',')
    rand = random.Random(args.seed)

    if not os.path.exists(os.path.join(args.dest, args.predir, "total")):
        os.makedirs(os.path.join(args.dest, args.predir, "total"))
    
    with open(os.path.join(args.dest, args.predir, "total/train.tsv"), "w") as total_f:
        print(root_path, file=total_f)

    for s in subset:
        if not os.path.exists(os.path.join(args.dest, args.predir, s.lower())):
            os.makedirs(os.path.join(args.dest, args.predir, s.lower()))

        dir_path = os.path.join(args.root, s)
        search_path = os.path.join(dir_path, "**/*." + args.ext)

        with open(os.path.join(args.dest, args.predir, s.lower(), "train.tsv"), "w") as train_f, open(
            os.path.join(args.dest, args.predir, s.lower(), "valid.tsv"), "w") as valid_f, open(
            os.path.join(args.dest, args.predir, s.lower(), "test.tsv"), "w") as test_f, open(
            os.path.join(args.dest, args.predir, "total/train.tsv"), "a"
            ) as total_f:
            print(dir_path, file=train_f)
            print(dir_path, file=valid_f)
            print(dir_path, file=test_f)

            fnames = list(glob.iglob(search_path, recursive=True))
            rand.shuffle(fnames)
            
            valid_len = int(len(fnames) * args.valid_percent)
            test_len = int(len(fnames) * args.valid_percent)
            train_len = len(fnames) - (valid_len + test_len)

            train = fnames[:train_len]
            valid = fnames[train_len:train_len + valid_len]
            test = fnames[train_len + valid_len:]

            def write(fnames, dest):
                for fname in fnames:
                    file_path = os.path.realpath(fname)

                    if args.path_must_contain and args.path_must_contain not in file_path:
                        continue

                    print(
                        "{}".format(os.path.relpath(file_path, dir_path)), file=dest, end='\t'
                    )

                    if "train" in dest.name or (
                        "valid" in dest.name
                    ):
                        print(
                            "{}".format(os.path.relpath(file_path, root_path)), file=total_f, end='\t'
                        )

                    if args.ext == 'mat':
                        import scipy.io
                        data = scipy.io.loadmat(file_path)
                        #NOTE you should preprocess data to match with given keys: "feats", "curr_sample_rate", "label"
                        length = data['feats'].shape[-1]
                        print(length, file=dest)

                        if "train" in dest.name or (
                            "valid" in dest.name
                        ):
                            print(length, file=total_f)

            write(train, train_f)
            write(valid, valid_f)
            write(test, test_f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
