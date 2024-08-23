import argparse
import glob
import os

"""
Combine two manifests based on their filenames.
Given two directories, it tries to find .tsv files that have the same filename within both of the
two directories, and combine them. For example, if one has "train.tsv" and "test.tsv" and the other
has "train.tsv", "valid.tsv", and "test.tsv", it outputs "train.tsv" and "test.tsv" by combining
each of the common manifest files (train.tsv and test.tsv).
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("first", metavar="DIR",
                       help="first root directory containing manifest (.tsv) files to be combined")
    parser.add_argument("second", metavar="DIR",
                       help="second root directory containing manifest (.tsv) files to be combined")

    parser.add_argument(
        "--exclude",
        default=None,
        type=str,
        help="filename to be excluded from the combined targets. if not provided, combine all the "
            "common manifest files within both of the two root directories"
    )
    parser.add_argument(
        "--dest",
        type=str,
        metavar="DIR",
        help="output directory"
    )

    return parser

def main(args):
    dest_path = os.path.realpath(args.dest)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    first_manifests = glob.glob(os.path.join(args.first, "*.tsv"))
    second_manifests = glob.glob(os.path.join(args.second, "*.tsv"))

    common_manifests = []
    for x in first_manifests:
        xname = os.path.basename(x)
        for y in second_manifests:
            yname = os.path.basename(y)
            if xname == yname:
                common_manifests.append((x, y))
                break

    for x, y in common_manifests:
        with open(x, "r") as f:
            first_root = f.readline().strip()
            first_lines = f.readlines()
        with open(y, "r") as f:
            second_root = f.readline().strip()
            second_lines = f.readlines()
        common_root = os.path.commonpath([first_root, second_root])
        if common_root == "":
            common_root = "/"
        first_relpath = os.path.relpath(first_root, common_root) + "/"
        second_relpath = os.path.relpath(second_root, common_root) + "/"

        with open(os.path.join(dest_path, os.path.basename(x)), "w") as f:
            print(common_root, file=f)
            for line in first_lines:
                line = line.strip()
                print(first_relpath + line, file=f)
            for line in second_lines:
                line = line.strip()
                print(second_relpath + line, file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)