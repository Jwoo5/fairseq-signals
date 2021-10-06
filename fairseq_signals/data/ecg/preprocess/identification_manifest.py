import argparse
import glob
import os
import random

import scipy.io

"""
Index .mat files contained in root directory to perform identification task

    Usage: python path/to/identification_manifest.py \
            /path/to/signals \
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
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation (between 0 and 0.5)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    root_path = os.path.realpath(args.root)
    search_path = os.path.join(args.root, "**/*." + args.ext)
    rand = random.Random(args.seed)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "valid_gallery.tsv"), "w") as valid_gallery_f, open(
        os.path.join(args.dest, "valid_probe.tsv"), "w" ) as valid_probe_f, open(
        os.path.join(args.dest, "test_gallery.tsv"), "w") as test_gallery_f, open(
        os.path.join(args.dest, "test_probe.tsv"), "w"
        ) as test_probe_f:
        print(root_path, file=train_f)
        print(root_path, file=valid_gallery_f)
        print(root_path, file=valid_probe_f)
        print(root_path, file=test_gallery_f)
        print(root_path, file=test_probe_f)

        idx = 0
        valid_idx = 0
        test_idx = 0
        patients = {}
        for fname in glob.iglob(search_path, recursive=True):
            data = scipy.io.loadmat(fname)
            patient_id = data['patient_id'][0]

            if patient_id in patients:
                patients[patient_id].append(fname)
            else:
                patients[patient_id] = [fname]

        for _, patient in patients.items():
            prob = rand.random()
            if prob <= args.valid_percent:
                if len(patient) < 2:
                    continue
                for i, p in enumerate(patient[:2]):
                    dest = valid_gallery_f if i % 2 == 0 else valid_probe_f
                    data = scipy.io.loadmat(p)
                    length = data['feats'].shape[-1]
                    print(
                        "{}\t{}\t{}".format(
                            os.path.relpath(p, root_path), length, valid_idx
                        ), file=dest
                    )
                valid_idx += 1
            elif args.valid_percent < prob <= 2 * args.valid_percent:
                if len(patient) < 2:
                    continue
                for i, p in enumerate(patient[:2]):
                    dest = test_gallery_f if i % 2 == 0 else test_probe_f
                    data = scipy.io.loadmat(p)
                    length = data['feats'].shape[-1]
                    print(
                        "{}\t{}\t{}".format(
                            os.path.relpath(p, root_path), length, test_idx
                        ), file=dest
                    )
                test_idx += 1
            else:
                for p in patient:
                    data = scipy.io.loadmat(p)
                    length = data['feats'].shape[-1]
                    print(
                        "{}\t{}\t{}".format(
                            os.path.relpath(p, root_path), length ,idx
                        ), file=train_f
                    )
                idx += 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)