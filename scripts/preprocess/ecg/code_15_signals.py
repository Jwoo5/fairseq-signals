"""Signal processing code specific to CODE-15% Version 1.0.0 (https://zenodo.org/records/4916206).

Has no record extraction code, simply rename exams.csv to records.csv and place it
in the processed root.
"""

import os

from scipy.io import savemat

import numpy as np
import pandas as pd

import h5py

from preprocess import (
    get_pipeline_parser,
    pipeline,
)
from preprocess import (
    extract_feat_info,
    reorder_leads,
)

SOURCE = "code_15"

def strip_zeros(arr: np.ndarray) -> np.ndarray:
    """
    Remove all-zero columns from the beginning and end of a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        2D array.

    Returns
    -------
    np.ndarray
        The 2D array with all-zero columns removed from the beginning and end.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    # Find indices of columns that are not entirely zero
    non_zero_columns = np.any(arr != 0, axis=0)
    nonzero_inds = np.where(non_zero_columns)[0]

    # If all zero, simply return without slicing (otherwise preprocessing will fail)
    if len(nonzero_inds) == 0:
        return arr

    # Otherwise, slice to exclude 
    return arr[:, nonzero_inds.min(): nonzero_inds.max() + 1]

def extract_code_15(row, leads_to_load):
    hdf_file = HDFs[row['trace_file']]
    feats = np.moveaxis(hdf_file['tracings'][row['hdf_ind']], 0, -1)

    # Strip leading and trailing zeros (padding added to fit the HDF5 format)
    feats = strip_zeros(feats)
    sample_size = feats.shape[-1]

    # Re-order leads
    feats, avail_leads = reorder_leads(
        feats,
        ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        leads_to_load,
    )

    fields = {
        "org_sample_rate": 400,
        "curr_sample_rate": 400,
        "org_sample_size": sample_size,
        "curr_sample_size": sample_size,
        "feats": feats,
        "idx": row["idx"],
    }
    savemat(row["save_path"], fields)

    meta = pd.concat([
        pd.Series({
            "sample_rate": fields["org_sample_rate"],
            "sample_size": fields["org_sample_size"],
            "avail_leads": str(avail_leads),
        }),
        pd.Series(extract_feat_info(feats, leads_to_load)),
    ])

    return meta

def main(args):
    global HDFs

    try:
        records = pd.read_csv(os.path.join(args.processed_root, "records.csv"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            'Rename exams.csv to records.csv and place it under `processed_root`.'
        ) from err

    records['path'] = records['exam_id'] # Need this to pass column checking
    records['save_file'] = 'code_15_' + records['exam_id'].astype(str) + '.mat'
    records["source"] = SOURCE
    records["dataset"] = SOURCE

    HDFs = {
        filename: h5py.File(os.path.join(args.raw_root, filename), 'r') \
        for filename in [f'exams_part{i}.hdf5' for i in range(18)]
    }

    # Note: last sample has non-existent exam_id 0 and has all-zero tracing
    exam_ids = {key: np.array(HDFs[key]['exam_id'][:])[:-1] for key in HDFs}

    exam_id_to_ind = {}
    for key in HDFs:
        exam_id_to_ind.update(dict(zip(exam_ids[key], np.arange(len(exam_ids[key])))))

    records['hdf_ind'] = records['exam_id'].map(exam_id_to_ind)

    pipeline(
        args,
        records,
        SOURCE,
        extract_func=extract_code_15,
    )

if __name__ == "__main__":
    parser = get_pipeline_parser()
    args = parser.parse_args()
    main(args)
