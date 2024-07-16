import logging
logging.basicConfig(level = logging.INFO)

from typing import Callable, Dict, List, Optional
import os
import argparse

from importlib.util import find_spec

import wfdb

import numpy as np
import pandas as pd

from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

from fairseq_signals.utils.file import extract_filename, filenames_from_paths
from fairseq_signals.utils.pandas import (
    check_cols,
    drop_na_cols,
    explode_with_order,
    numpy_series_to_dataframe,
)

LEAD_ORDER = np.array(
    ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
)
LEAD_TO_IND = {lead: lead_ind for lead_ind, lead in enumerate(LEAD_ORDER)}
IND_TO_LEAD = {lead_ind: lead for lead_ind, lead in enumerate(LEAD_ORDER)}
STDs = np.array([
    [0.12781512],
    [0.14941794],
    [0.12525026],
    [0.12572662],
    [0.10236586],
    [0.12168122],
    [0.1870286 ],
    [0.28818728],
    [0.29218111],
    [0.29028273],
    [0.26973001],
    [0.23605948],
])

FEMALE_VALUE = 1
MALE_VALUE = 0

def resample(feats, curr_sample_rate, desired_sample_rate):
    """
    Resample an ECG using linear interpolation.
    """
    if curr_sample_rate == desired_sample_rate:
        return feats

    desired_sample_size = int(
        feats.shape[-1] * (desired_sample_rate / curr_sample_rate)
    )

    x = np.linspace(0, desired_sample_size - 1, feats.shape[-1])

    return interp1d(x, feats, kind='linear')(np.arange(desired_sample_size))

def reorder_leads(feats: np.ndarray, sig_name: List[str], leads_to_load: pd.DataFrame):
    sig_name = np.array(sig_name)

    # If already identical, simply return feats as is
    if np.array_equal(leads_to_load.index, sig_name):
        return feats, leads_to_load.index.values.tolist()

    feats_order = leads_to_load.join(
        pd.Series(np.arange(len(sig_name)), index=sig_name, name='sample_order'),
        how='left',
    )

    lead_missing = feats_order['sample_order'].isna()

    # If no missing leads, simply re-order the leads
    if not lead_missing.any():
        feats = feats[feats_order['sample_order'].astype(int)]
        return feats, leads_to_load.index.values.tolist()

    # Otherwise, create a whole new array and fill in the available leads
    feats_new = np.full((len(leads_to_load), feats.shape[1]), np.nan)

    avail = feats_order[~lead_missing].astype(int)
    for _, row in avail.iterrows():
        feats_new[avail['global_order']] = feats[avail['sample_order']]

    return feats_new, lead_missing.index[~lead_missing].values.tolist()

def extract_feat_info(feats: np.ndarray, leads_to_load):
    fields = {}

    mean = feats.mean(axis=1, keepdims=True)
    fields['mean'] = mean.flatten()
    fields['std'] = np.std(feats, axis=1)

    nan = np.isnan(feats)
    nan_lead_sum = nan.sum(axis=1)
    nan_leads = nan_lead_sum == feats.shape[1]
    fields['nan_any'] = nan_lead_sum.sum() > 0
    fields['nan_leads_any'] = nan_leads.any()
    fields["nan_leads"] = str(
        leads_to_load.index[nan_leads].to_list()
    )

    constant_leads = np.count_nonzero(feats - mean, axis=1) == 0
    fields['constant_leads_any'] = constant_leads.any()
    fields["constant_leads"] = str(
        leads_to_load.index[constant_leads].to_list()
    )

    return fields

def extract_wfdb(row, leads_to_load):
    feats, fields = wfdb.rdsamp(row["path"])

    # Re-order leads to a selected standard order
    feats = np.moveaxis(feats, 0, -1)
    feats, avail_leads = reorder_leads(feats, fields["sig_name"], leads_to_load)

    data = {
        "org_sample_rate": fields["fs"],
        "curr_sample_rate": fields["fs"],
        "org_sample_size": fields["sig_len"],
        "curr_sample_size": fields["sig_len"],
        "feats": feats,
        "idx": row["idx"],
    }

    savemat(row["save_path"], data)

    fields["avail_leads"] = str(avail_leads)
    fields.update(extract_feat_info(feats, leads_to_load))

    return pd.Series(fields)

def lead_std_divide(feats, constant_lead_strategy='zero'):
    # Calculate standard deviation along axis 1, keep dimensions for broadcasting
    std = feats.std(axis=1, keepdims=True)
    std_zero = std == 0

    # Check if there are any zero stds or if strategy is 'nan'
    if not std_zero.any() or constant_lead_strategy == 'nan':
        # Directly divide, which will turn constant leads into NaN if any
        feats = feats / std

        return feats, std

    # Replace zero standard deviations with 1 temporarily to avoid division by zero
    std_replaced = np.where(std_zero, 1, std)
    feats = feats / std_replaced

    if constant_lead_strategy == 'zero':
        # Replace constant leads to be 0
        zero_mask = np.broadcast_to(std_zero, feats.shape)
        feats[zero_mask] = 0

    elif constant_lead_strategy == 'constant':
        # Leave constant leads as is
        pass

    else:
        raise ValueError("Unexpected constant lead strategy.")

    return feats, std

def preprocess_mat(row, desired_sample_rate, standardize, constant_lead_strategy):
    data = loadmat(row["path"])
    feats = data["feats"]
    curr_sample_rate = data["curr_sample_rate"][0, 0]

    del data["__header__"]
    del data["__version__"]
    del data["__globals__"]

    # Resample
    feats = resample(feats, curr_sample_rate, desired_sample_rate)

    # Standardize
    if standardize:
        mean = feats.mean(axis=1, keepdims=True)
        feats = feats - mean
        feats, std = lead_std_divide(
            feats,
            constant_lead_strategy=constant_lead_strategy,
        )

    data["feats"] = feats
    data["curr_sample_rate"] = desired_sample_rate
    data["curr_sample_size"] = feats.shape[-1]
    data["mean"] = mean
    data["std"] = std

    savemat(row["save_path"], data)

def segment_mat(row, seconds, expected_sample_rate):
    data = loadmat(row["path"])
    feats = data["feats"]
    curr_sample_rate = data["curr_sample_rate"][0, 0]

    del data["__header__"]
    del data["__version__"]
    del data["__globals__"]

    if curr_sample_rate != expected_sample_rate:
        raise ValueError(
            f"Current sample rate {curr_sample_rate} does not match "
            f"expected sample rate {expected_sample_rate}."
        )

    # Segment
    segment_size = seconds * expected_sample_rate
    n_segments = feats.shape[-1] // segment_size

    data["curr_sample_size"] = segment_size
    save_files = []
    for segment_i in range(n_segments):
        segment_feats = \
            feats[:, segment_i * segment_size: (segment_i + 1) * segment_size]

        data["feats"] = segment_feats
        data["segment_i"] = segment_i

        save_file = row["save_path"][:-4] + f"_{segment_i}.mat"
        savemat(save_file, data)

        save_files.append(save_file)

    return save_files

def process_str_list_arg(text_lst):
    return pd.Series(text_lst).str.split(',').explode().str.strip()

def concat_extracted(extracted):
    """
    Concatenate return values (dictionary of DataFrames) from the .apply function,
    preserving the original index of each DataFrame.

    Parameters
    ----------
    extracted : pd.Series
        Series where each element is a dictionary of DataFrames.

    Returns
    -------
    dict of pd.DataFrame
        Dictionary where keys are the original dictionary keys present in the results,
        and values are DataFrames resulting from concatenating all DataFrames under
        each key across all elements of the Series, preserving the original indexes.

    Raises
    ------
    ValueError
        If the extracted Series is empty.
    """
    # Check if the results series is empty
    if extracted.empty:
        raise ValueError("The extracted series is empty. Check the input Series.")

    # Assume the first item's keys represent all possible keys
    all_keys = extracted.iloc[0].keys()

    concatenated = {}
    for key in all_keys:
        dfs = extracted.apply(lambda x: x[key])
        concatenated[key] = pd.concat(dfs.tolist())

    return concatenated

def get_pipeline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="Path to save the processed data root directory.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Path to the raw data root directory.",
    )
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to multi-source manifest. If specified but nonexistent, then "
            "created, otherwise appended. Necessary to maintain unique sample idx "
            "across multiple data sources. Do not specify when debugging.",
    )
    parser.add_argument(
        "--results_suffix",
        type=str,
        default='',
        help="Suffix used for results files. Useful when processing several versions.",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        help="Comma-delimited list of datasets to process.",
    )
    parser.add_argument(
        "--desired_sample_rate",
        type=int,
        default=500,
        help="Desired sample rate.",
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=5,
        help="Desired segment size in seconds.",
    )
    parser.add_argument(
        "--no_standardization",
        action="store_true", 
        help="Skip standardizing signals to have mean 0 and STD 1.",
    )
    parser.add_argument(
        "--leads",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        type=str,
        help="Comma-delimited lead numbers, e.g., '0,1' loads leads I and II. "
        "The order is as follows: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
    )
    parser.add_argument(
        "--constant_lead_strategy",
        default="zero",
        type=str,
        help="Strategy to handle constant leads. Options are 'zero', 'nan', and "
        "'constant'. Default 'zero' since random lead masking masks with 0s and "
        "missing leads could be treated as masked."
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-delimited list of steps to skip, e.g., 'org,preprocessed'. "
        "Specified out of 'org', 'preprocessed', and 'segmented'."
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true", 
        help="Prevents parallel processing. Must install pandarallel if unspecified.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=None,
        help="Number of pandarallel workers. Defaults to number of available cores.",
    )

    return parser

def pipeline(
    args: argparse.Namespace,
    records: pd.DataFrame,
    source: str,
    extract_func: Callable = extract_wfdb,
    postprocess_extraction: Optional[Dict[str, Callable]] = None,
):
    """
    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from `get_pipeline_parser`.
    records : pandas.DataFrame
        Records data. Must have 'path', 'source', and 'dataset' columns.
    source : str
        Name of the data source being processed.
    extract_func : callable
        Function to extract ECG sample metadata and save a corresponding .mat file.
    postprocess_extraction : dict of callable, optional
        Optional functions to post-process the output(s) of extract_func.
    """
    check_cols(records, ["path", "source", "dataset"], raise_err_on_missing=True)
    check_cols(
        records,
        [
            "idx",
            "save_path",
            "source_path",
            "org_path",
            "preprocessed_path",
            "segmented_path",
        ],
        raise_err_on_existing=True,
    )
    args.skip = [] if args.skip is None else args.skip

    # Create unique sample IDs
    records.index.name = "idx"
    records.reset_index(inplace=True)
    if args.manifest_file is None:
        logging.info('No multi-source manifest specified. Be cautious of duplicate sample idx.')
    elif os.path.isfile(args.manifest_file):
        manifest = pd.read_csv(args.manifest_file)
        records["idx"] += manifest["idx"].max() + 1

    if args.no_parallel:
        from tqdm import tqdm
        tqdm.pandas()
        apply_method = 'progress_apply'
    else:
        if find_spec('pandarallel') is None:
            raise ValueError("Please install pandarallel or specify `--no_parallel`.")

        from pandarallel import pandarallel
        if args.nb_workers is None:
            kwargs = {}
        else:
            kwargs = {'nb_workers': args.nb_workers}

        pandarallel.initialize(progress_bar=True, **kwargs)

        apply_method = 'parallel_apply'

    if args.dataset_subset is not None:
        args.dataset_subset = process_str_list_arg(args.dataset_subset).values
        records = records[records['dataset'].isin(args.dataset_subset)].copy()

    # Prepare directories
    args.processed_root = args.processed_root.rstrip("/") + "/"
    org_dir = args.processed_root + "org/"
    preprocessed_dir = args.processed_root + "preprocessed/"
    segmented_dir = args.processed_root + "segmented/"

    for directory in [org_dir, preprocessed_dir, segmented_dir]:
        os.makedirs(directory, exist_ok=True)

    # Prepare save paths
    meta = records
    if "save_file" not in meta.columns:
        meta["save_file"] = filenames_from_paths(meta["path"], replacement_ext=".mat")

    assert meta["save_file"].is_unique

    meta["save_path"] = org_dir + meta["save_file"]

    if "org" not in args.skip:
        logging.info("Extracting data...")

        if not meta["save_path"].is_unique:
            raise ValueError("Save paths are not unique.")

        # Determine leads to load
        leads_to_load = process_str_list_arg(args.leads).astype(int)
        leads_to_load = leads_to_load.sort_values().drop_duplicates().map(IND_TO_LEAD)
        leads_to_load = pd.DataFrame(
            {'global_order': np.arange(len(leads_to_load))},
            index=leads_to_load,
        )

        if leads_to_load.index.isna().any():
            raise ValueError('Invalid leads.')

        logging.info(f"Saving leads with order: {', '.join(leads_to_load.index)}")
        meta['lead_order'] = str(leads_to_load.index.to_list())

        extracted = getattr(meta, apply_method)(
            extract_func,
            args=(leads_to_load,),
            axis=1,
        )

        # Standardize format (whether multiple DataFrames returned or only meta)
        if isinstance(extracted, pd.DataFrame):
            extracted = {'meta': extracted}
        else:
            extracted = concat_extracted(extracted)
            assert 'meta' in extracted

        extracted['meta'] = pd.concat(
            [extracted['meta'], numpy_series_to_dataframe(extracted['meta']['mean'])],
            axis=1,
        )
        extracted['meta'] = pd.concat(
            [extracted['meta'], numpy_series_to_dataframe(extracted['meta']['std'])],
            axis=1,
        )
        extracted['meta'].drop(['mean', 'std'], axis=1, inplace=True)
        extracted['meta'] = pd.concat([meta, extracted['meta']], axis=1)

        if postprocess_extraction is not None:
            for key, postprocess_fn in postprocess_extraction.items():
                extracted[key] = postprocess_fn(extracted[key])

        meta = extracted['meta']

        for key, data in extracted.items():
            data['save_file'] = data.index.map(meta['save_file'].to_dict())
            data['idx'] = data.index.map(meta['idx'].to_dict())
            data.to_csv(os.path.join(args.processed_root, f'{key}{args.results_suffix}.csv'), index=False)

    # Preprocess data
    meta.rename({"path": "source_path", "save_path": "org_path"}, axis=1, inplace=True)
    meta["path"] = meta["org_path"]
    meta["save_path"] = preprocessed_dir + meta["save_file"]

    if "preprocessed" not in args.skip:
        logging.info("Preprocessing data...")
        getattr(meta, apply_method)(
            preprocess_mat,
            args=(args.desired_sample_rate, not args.no_standardization, args.constant_lead_strategy),
            axis=1,
        )

    # Segment data
    meta = meta.rename({"save_path": "preprocessed_path"}, axis=1)
    meta["path"] = meta["preprocessed_path"]
    meta["save_path"] = segmented_dir + meta["save_file"]

    if "segmented" not in args.skip:
        logging.info("Segmenting data...")
        segmented = getattr(meta, apply_method)(
            segment_mat,
            args=(args.segment_seconds, args.desired_sample_rate),
            axis=1,
        )

        segmented = segmented.explode().rename('path').sort_values().to_frame()
        segmented['save_file'] = extract_filename(
            segmented['path']
        ).str.replace(r'(_\d+\.mat)$', '.mat', regex=True)
        segmented['sample_size'] = int(args.desired_sample_rate*args.segment_seconds)
        segmented.to_csv(os.path.join(args.processed_root, f'segmented{args.results_suffix}.csv'), index=False)
    else:
        segmented = None

    meta = meta.rename({"save_path": "segmented_path"}, axis=1)
    meta = meta.drop("path", axis=1)

    # Create/update manifest file if specified
    if args.manifest_file is not None:
        manifest_meta = meta[
            ["idx", "source", "source_path", "save_file", "sample_rate", "sample_size"]
        ].copy()
        manifest_meta.to_csv(
            args.manifest_file,
            index=False,
            mode='a',
            header=not os.path.exists(args.manifest_file),
        )

def postprocess_wfdb(meta):
    # Drop any columns with all null values
    meta = drop_na_cols(meta)

    # If all the same unit (all samples, all leads), then standardize to one unit value
    # per sample
    units = explode_with_order(meta["units"])
    unit_vcs = units["units"].str.lower().value_counts()

    if len(unit_vcs) == 1:
        unit = unit_vcs.index[0]
        meta["units"] = unit

    meta = meta.drop(["sig_name", "n_sig"], axis=1)
    meta = meta.rename({
        "fs": "sample_rate",
        "sig_len": "sample_size",
    }, axis=1)

    return meta

def load_cmsc_manifest(filename):
    file = open(filename, "r").read()
    manifest = pd.Series([file]).str.split("\n").explode().iloc[2:-1]
    manifest = manifest.str.split('\t', expand=True)
    manifest.drop(2, axis=1, inplace=True)
    manifest.rename({0: 'path', 1: 'sample_size', 3: 'pair_segment_ids'}, axis=1, inplace=True)

    manifest["save_file"] = extract_filename(manifest["path"])

    return manifest
