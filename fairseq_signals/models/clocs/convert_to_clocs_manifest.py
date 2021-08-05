import argparse
import glob
import os
import random
from collections import Counter

"""

    Usage: python examples/clocs/convert_to_clocs_manifest.py \
            /path/to/manifest.tsv \
            --dest /manifest/path \
            --predir /sub/root/dir \
            --ext $ext \
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="manifest file path to convert, "
             "should be consistent with tsv format"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--predir", default=".", type=str, metavar="DIR", help="if set, create sub-root directory in --dest"
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT",
        help="extension of data files in the manifest"
    )
    return parser

def main(args):
    fnames = []
    sizes = {}
    with open(args.root, "r") as f:
        dir_path = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert len(items) == 2, line
            fnames.append(items[0])
            sizes[items[0][:items[0].rindex("_")]] = items[1]
    num_segs = dict(Counter([f[:f.rindex("_")] for f in fnames]))

    if not os.path.exists(os.path.join(args.dest, args.predir)):
        os.makedirs(os.path.join(args.dest, args.predir))

    with open(os.path.join(args.dest, args.predir, os.path.basename(args.root)), "w") as f:
        print(dir_path, file=f)
        print(args.ext, file=f)

        for fname, n_segs in num_segs.items():
            if n_segs <= 1:
                continue
            print(f"{fname}\t{sizes[fname]}\t{n_segs}", file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
