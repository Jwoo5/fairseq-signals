import argparse
import glob
import os
import random
from collections import Counter

"""

    Usage: python examples/clocs/convert_to_clocs_manifest.py \
            /path/to/manifest.tsv \
            --dest /manifest/path \
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
        "--ext", default="mat", type=str, metavar="EXT",
        help="extension of data files in the manifest"
    )
    parser.add_argument(
        "--n-lead", default=12, type=int, metavar="D",
        help="number of leads in each sample"
    )
    return parser

def main(args):
    fnames = []
    sizes = {}
    segments = {}
    with open(args.root, "r") as f:
        dir_path = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert len(items) == 2, line
            fnames.append(items[0])
            #TODO should aggregate over patient_id, not file names
            folder = items[0][:items[0].rindex("_")]
            sizes[folder] = items[1]

            segment = int(items[0][items[0].rindex("_") + 1:-4])
            if folder in segments:
                segments[folder].append(segment)
            else:
                segments[folder] = [segment]

    if not os.path.exists(os.path.join(args.dest, "cmsc")):
        os.makedirs(os.path.join(args.dest, "cmsc"))
    if not os.path.exists(os.path.join(args.dest, "cmlc")):
        os.makedirs(os.path.join(args.dest, "cmlc"))
    if not os.path.exists(os.path.join(args.dest, "cmsmlc")):
        os.makedirs(os.path.join(args.dest, "cmsmlc"))
    

    with open(os.path.join(args.dest, "cmsc", "train.tsv"), "w") as cmsc_f, open(
        os.path.join(args.dest, "cmlc", "train.tsv"), "w") as cmlc_f, open(
        os.path.join(args.dest, "cmsmlc", "train.tsv"), "w"
        ) as cmsmlc_f:
        print(dir_path, file=cmsc_f)
        print(dir_path, file=cmlc_f)
        print(dir_path, file=cmsmlc_f)
        print(args.ext, file=cmsc_f)
        print(args.ext, file=cmlc_f)
        print(args.ext, file=cmsmlc_f)

        leads = list(range(args.n_lead))
        for fname, segment in segments.items():
            n_segs = len(segment)
            if n_segs <= 1:
                continue
            segment.sort()

            for i in range(0, len(segment)-1, 2):
                seg = ','.join(str(seg) for seg in segment[i:i+2])
                print(f"{fname}\t{sizes[fname]}\t0\t{seg}", file=cmsmlc_f)
                for lead in leads:
                    print(f"{fname}\t{sizes[fname]}\t{lead}\t{seg}", file=cmsc_f)

            for seg in segment:
                print(f"{fname}\t{sizes[fname]}\t0\t{seg}", file=cmlc_f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
