"""Signal processing code specific to PhysioNet 2021 v1.0.3."""

import os

import numpy as np
import pandas as pd

from preprocess import (
    FEMALE_VALUE,
    MALE_VALUE,
    get_pipeline_parser,
    pipeline,
    postprocess_wfdb,
)
from fairseq_signals.utils.file import filenames_from_paths

SOURCE = "physionet2021"

def physionet2021_comments(comments):
    # Non-greedy pattern to extract the different components from the comments
    comment_fields = comments.astype(str).str.extractall(r"'(.*?:[^']*)',")
    comment_fields = pd.concat([
        comment_fields,
        comment_fields[0].str.split(":", regex=False, expand=True).rename(
            {0: "field", 1: "value"},
            axis=1
        )
    ], axis=1)
    comment_fields = comment_fields.drop(0, axis=1)
    comment_fields.index = comment_fields.index.droplevel(1)

    # Process values
    comment_fields["value"] = comment_fields["value"].str.strip("', ").replace("Unknown", np.nan)

    # Pivot to turn the rows into columns
    comment_fields = comment_fields.reset_index().pivot_table(
        index='index',
        columns='field',
        values='value',
        aggfunc='first',
        dropna=True,
    )
    comment_fields.index.name = None

    comment_fields = comment_fields.rename({"Age": "age", "Dx": "diagnosis", "Sex": "sex"}, axis=1)

    # Process sex
    comment_fields["sex"] = comment_fields["sex"].map({"Female": FEMALE_VALUE, "Male": MALE_VALUE}).astype("Int64")

    comment_fields.columns.name = None

    return comment_fields

def postprocess_physionet2021(meta):
    meta = postprocess_wfdb(meta)

    # Process comments
    meta = pd.concat([
        meta.drop("comments", axis=1),
        physionet2021_comments(meta["comments"]),
    ], axis=1)
    
    return meta

def main(args):
    if os.path.normpath(args.raw_root).split(os.sep)[-1] != 'training':
        args.raw_root = os.path.join(args.raw_root, 'training')

    records = pd.read_csv(os.path.join(args.processed_root, "records.csv"))
    records["path"] = args.raw_root.rstrip('/') + '/' + records["path"]
    records["save_file"] = (SOURCE + "_") + \
        filenames_from_paths(records["path"], replacement_ext=".mat")
    records["source"] = SOURCE

    pipeline(
        args,
        records,
        SOURCE,
        postprocess_extraction={'meta': postprocess_physionet2021},
    )

if __name__ == "__main__":
    parser = get_pipeline_parser()
    args = parser.parse_args()
    main(args)
