import argparse
import glob
import os
import random
from collections import Counter

"""

    Usage: python /path/to/convert_to_cmsc_manifest.py \
            /path/to/manifest \
            --dest /path/to/converted/manifest \
            --ext $ext \
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="directory path containing manifest files to convert"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT",
        help="extension of data files in the manifest"
    )
    parser.add_argument(
        "--n-lead", default=12, type=int, metavar="D",
        help="number of leads in each sample"
    )
    return parser

def main(args):
    manifests_tbc = glob.glob(os.path.join(args.root, "*.tsv"))
    if not os.path.exists(os.path.join(args.dest, "cmsc")):
        os.makedirs(os.path.join(args.dest, "cmsc"))

    for manifest in manifests_tbc:
        fnames = []
        sizes = {}
        segments = {}

        with open(manifest, "r") as f:
            dir_path = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, (
                    f"{manifest} does not follow the appropriate format, "
                    f"(e.g., {line})"
                )
                fnames.append(items[0])
                # collect data segments from the same file
                folder = items[0][:items[0].rindex("_")]
                sizes[folder] = items[1]
                
                segment = int(items[0][items[0].rindex("_") + 1:-4])
                if folder in segments:
                    segments[folder].append(segment)
                else:
                    segments[folder] = [segment]

        with open(os.path.join(args.dest, "cmsc", os.path.basename(manifest)), "w") as f:
            print(dir_path, file=f)
            print(args.ext, file=f)

            for fname, segment in segments.items():
                n_segs = len(segment)
                if n_segs <= 1:
                    continue
                segment.sort()
                
                for i in range(0, len(segment) - 1, 2):
                    seg = ",".join(str(seg) for seg in segment[i:i+2])
                    print(f"{fname}\t{sizes[fname]}\t0\t{seg}", file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
