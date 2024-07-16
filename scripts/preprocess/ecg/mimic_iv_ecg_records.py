"""Record extraction code specific to MIMIC-IV-ECG v1.0 (and optionally MIMIC-IV v2.2)."""

import os
import argparse

import pandas as pd

from preprocess import FEMALE_VALUE, MALE_VALUE

def get_parser():
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
        help="Path to the MIMIC-IV-ECG root directory.",
    )
    parser.add_argument(
        "--mimic_iv_root",
        type=str,
        required=False,
        help="Path to the MIMIC-IV root directory (optional).",
    )
    
    return parser

def process_patients(path: str) -> pd.DataFrame:
    """
    Processes the patient dataset and performs data manipulations.

    Parameters
    ----------
    path : str
        The file path to the patient CSV data.

    Returns
    -------
    pd.DataFrame
        The processed patient data.
    """
    patients = pd.read_csv(path)

    patients.rename({'gender': 'sex'}, axis=1, inplace=True)
    patients["sex"] = patients["sex"].map({"F": FEMALE_VALUE, "M": MALE_VALUE}).astype("Int64")

    assert ((patients['anchor_year_group'].str.slice(stop=4).astype(int) + 2) == 
            patients['anchor_year_group'].str.slice(start=-4).astype(int)).all()

    patients['anchor_year_group_middle'] = patients['anchor_year_group'].str.slice(stop=4).astype(int) + 1
    patients['anchor_year_offset'] = patients['anchor_year_group_middle'] - patients['anchor_year']
    patients['anchor_day_offset'] = (patients['anchor_year_offset'] * 365.25).astype('timedelta64[D]')
    patients['dod_anchored'] = pd.to_datetime(patients['dod']) + patients['anchor_day_offset']
    patients.drop(['anchor_year', 'anchor_year_group', 'anchor_year_offset', 'dod'], axis=1, inplace=True)
    patients.rename({'anchor_year_group_middle': 'anchor_year', 'dod_anchored': 'dod'}, axis=1, inplace=True)
    patients['anchor_year_dt'] = pd.to_datetime({'year': patients['anchor_year'], 'month': 1, 'day': 1})

    return patients

def process_cardiac_markers(path: str) -> pd.DataFrame:
    """
    Processes the cardiac marker data from a CSV file.

    Parameters
    ----------
    path : str
        The file path to the cardiac markers CSV data.

    Returns
    -------
    pd.DataFrame
        The processed cardiac markers data.
    """
    cardiac_marker_chunks = []
    for chunk in pd.read_csv(path, chunksize=1e5, low_memory=False):
        filtered_chunk = chunk[
            chunk['itemid'].isin([
                # 51002, # Troponin I (troponin-I is not measured in MIMIC-IV)
                # 52598, # Troponin I, point of care, rare/poor quality
                51003, # Troponin T
                50911, # Creatinine Kinase, MB isoenzyme
            ])
        ]
        cardiac_marker_chunks.append(filtered_chunk)

    cardiac_markers = pd.concat(cardiac_marker_chunks)
    cardiac_markers['itemid'].map({51003: 'Troponin T', 50911: 'Creatine Kinase, MB Isoenzyme'})

    return cardiac_markers

def main(args):
    """
    Main processing function that combines and manipulates multiple datasets.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    os.makedirs(args.processed_root, exist_ok=True)
    
    record_list_csv = os.path.join(
        args.raw_root,
        "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv",
    )
    records = pd.read_csv(record_list_csv)

    results = {}
    if args.mimic_iv_root:
        # Process patients
        patients = process_patients(
            os.path.join(args.mimic_iv_root, "hosp/patients.csv.gz")
        )
        results['patients'] = patients
        
        # Update acquisition times and ages based on anchor group
        records = records.merge(patients, how='left', on='subject_id')

        records['ecg_time'] = pd.to_datetime(records['ecg_time']) + \
            records['anchor_day_offset']

        records['age'] = records['anchor_age'] + \
            (records['ecg_time'] - records['anchor_year_dt']).dt.days / 365.25

        # Process cardiac markers
        cardiac_markers = process_cardiac_markers(
            os.path.join(args.mimic_iv_root, "hosp/labevents.csv.gz")
        )
        results['cardiac_markers'] = cardiac_markers

    # Incorporate machine diagnoses/reports
    machine_diagnoses = pd.read_csv(
        os.path.join(args.raw_root, "mimic_iv_ecg_machine_diagnoses.csv")
    )
    records['machine_diagnosis'] = machine_diagnoses['0']

    machine_report = pd.read_csv(
        os.path.join(args.raw_root, "mimic_iv_ecg_machine_report.csv")
    )
    records['machine_report'] = machine_report['0']
    results['records'] = records

    # Save results
    for filename, data in results.items():
        data.to_csv(os.path.join(args.processed_root, f"{filename}.csv"), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
