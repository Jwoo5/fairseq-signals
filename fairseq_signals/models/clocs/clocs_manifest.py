import argparse
import glob
import os
import random

"""
    Usage: python examples/clocs/clocs_manifest.py \
            /path/to/signals --dest /manifest/path \
            --ext $ext \
            --valid-percent $valid
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
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
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)
    checked = set()

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "valid.tsv"), "w"
    ) as valid_f:
        print(dir_path, file=train_f)
        print(args.ext, file=train_f)
        print(dir_path, file=valid_f)
        print(args.ext, file=valid_f)

        for fname in glob.iglob(search_path, recursive= True):
            file_path = os.path.realpath(fname)

            if (
                args.path_must_contain and args.path_must_contain not in file_path
            ) or (os.path.basename(fname[:fname.rindex("_")]) in checked):
                continue
            checked.add(os.path.basename(fname[:fname.rindex("_")]))

            dest = train_f if rand.random() > args.valid_percent else valid_f            

            #TODO handle files not in case of .mat
            if args.ext == 'mat':
                import scipy.io
                data = scipy.io.loadmat(file_path)

                file_path = os.path.realpath(fname[:fname.rindex("_")])
                print(
                    f"{os.path.relpath(file_path, dir_path)}", file=dest, end='\t'
                )

                #NOTE you should preprocess data to match with given keys: "feats", "curr_sample_rate", "label", "num_segs"
                length = data['feats'].shape[-1]
                num_segs = data["num_segs"][0,0]
                print(length, file = dest, end = '\t')
                print(num_segs, file = dest)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
