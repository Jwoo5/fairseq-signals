"""Signal processing code example and guide."""

from typing import Union
import os

import numpy as np
import pandas as pd

from preprocess import (
    get_pipeline_parser,
    pipeline,
)
from fairseq_signals.utils.file import filenames_from_paths

SOURCE = "example_source"

def extract_func(
    row: pd.Series,
    leads_to_load: pd.DataFrame,
) -> Union[pd.Series, pd.DataFrame, dict]:
    """
    Extract ECG sample metadata and save a standardized .mat file.

    Parameters
    ----------
    row : pandas.Series
        Row of records data.
    leads_to_load : pandas.DataFrame
        Ordered leads to load which is used in the `reorder_leads` function.

    Return
    ------
    pandas.Series or pandas.DataFrame or dict of pandas.DataFrame
        A Series representing extracted metadata, or optionally a dictionary of
        Series/DataFrames which must contain a 'meta' entry.
    """
    # TODO - Use `row["path"]` to load the sample

    fields = {
        'sample_rate': ... # TODO - Extract sample rate from the row or the file
    }

    feats = ... # A NumPy array having shape (channels, sample size)

    fields["sig_name"] = ... # A list of signal names representing the channel order
    # Must be a subset of: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Re-order leads and handle missing
    feats, avail_leads = reorder_leads(feats, sig_name, leads_to_load)

    # Define and save a standard .mat file
    # Other information may be included, but these fields must be present
    data = {
        "org_sample_rate": fields["sample_rate"],
        "curr_sample_rate": fields["sample_rate"],
        "org_sample_size": feats.shape[1],
        "curr_sample_size": feats.shape[1],
        "feats": feats,
        "idx": row["idx"],
    }
    savemat(row["save_path"], data)

    fields["avail_leads"] = str(avail_leads)

    # Recommended to extract feature information on mean, STD, nulls, and constants
    fields.update(extract_feat_info(feats, leads_to_load))

    # Return the extracted metadata
    meta = pd.Series(fields)

    return meta

    # Optionally, multiple DataFrame objects can be returned in a dictionary format,
    # which is useful for extracting data with multiple entries per sample - if done
    # this way, there must be a 'meta' entry
    qrs_times = ...

    return {'meta': meta, 'qrs_times': qrs_times}

def postprocess_meta(meta):
    """
    Perform operations on the combined meta extracted using the `extract_func`.

    This may include dropping or renaming columns, mapping values, or performing
    other such transformations.
    """
    # TODO - Optional (can remove `postprocess_extraction` argument if unneeded)

    return meta

def main(args):
    # Load the records
    records = pd.read_csv(os.path.join(args.processed_root, "records.csv"))

    # If the "path" column contains partial paths, the paths must be joined with the
    # raw data directory root
    records["path"] = os.path.join(args.raw_root, records["path"])

    # Turn file paths into unique file names
    # (Adding in the source string will ensure file names are unique across sources)
    records["save_file"] = (SOURCE + "_") + \
        filenames_from_paths(records["path"], replacement_ext=".mat")

    records["source"] = SOURCE

    # Specify like so if source has one dataset, as this column must be present
    records["dataset"] = SOURCE

    pipeline(
        args,
        records,
        SOURCE,
        extract_func=extract_func,
        postprocess_extraction={'meta': postprocess_meta},
    )

if __name__ == "__main__":
    parser = get_pipeline_parser()
    args = parser.parse_args()
    main(args)
