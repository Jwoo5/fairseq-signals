"""Signal processing code specific to extracting MUSE v8.0.2.10132 ECGs."""

from typing import Dict
from collections import OrderedDict
import os
import re
import logging

import array
import base64

import xmltodict
from xml.parsers.expat import ExpatError
from lxml import etree

from scipy.io import savemat

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from preprocess import get_pipeline_parser, pipeline
from preprocess import FEMALE_VALUE, MALE_VALUE
from preprocess import (
    LEAD_ORDER,
    extract_feat_info,
    reorder_leads,
)
from fairseq_signals.utils.file import filenames_from_paths

LEAD_ORDER_SET = set(LEAD_ORDER)
MUSE_FIELDS = [
    'MuseInfo',
    'PatientDemographics',
    'TestDemographics',
    'Order',
    'RestingECGMeasurements',
    'OriginalRestingECGMeasurements',
    'Diagnosis',
    'OriginalDiagnosis',
    'ExtraQuestions',
    'QRSTimesTypes',
    'Waveform',
    'PharmaData',
]

def xml_to_dict(file: str) -> dict:
    """Parse an XML file into a Python dictionary.

    It is possible to recover from invalid '&', '<', and '>' tokens.

    Parameters
    ----------
    file : str
        The path to the XML file.

    Returns
    -------
    dict
        A dictionary representing the parsed XML, or None if there was a parsing error.
    """
    xml_string = open(file, "r").read().replace("&", "&#38;")

    try:
        return xmltodict.parse(xml_string)
    except ExpatError:
        pass

    # Recover from invalid tokens such as '<'
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(xml_string.encode('utf-8'), parser=parser)
    
    return xmltodict.parse(
        etree.tostring(root, method='xml', encoding='utf-8')
    )

def process_diagnosis(raw) -> str:
    if isinstance(raw, dict):
        raw = [raw]
    
    text = ""
    for dictionary in raw:
        if "StmtText" in dictionary and dictionary["StmtText"] is not None:
            text += dictionary["StmtText"] + " "

    # Fix spaces
    text = re.sub(' +', ' ', text.strip())
    
    return text

def augment_leads(leads: Dict[str, np.ndarray]):
    # Must have leads I and II
    if "I" not in leads or "II" not in leads:
        return leads

    # Einthoven's law
    # III = II - I
    if "III" not in leads:
        leads["III"] = np.subtract(leads["II"], leads["I"])

    # Goldberger's equations
    # aVR = (I + II) / -2
    if "aVR" not in leads:
        leads["aVR"] = np.add(leads["I"], leads["II"]) / -2

    # aVL = (I - III) / 2
    # aVL = I - II/2 (equivalent)
    if "aVL" not in leads:
        leads["aVL"] = np.subtract(leads["I"], leads["III"]) / 2

    # aVF = (II + III) / 2
    # aVF = -I/2 + II (equivalent)
    if "aVF" not in leads:
        leads["aVF"] = np.add(leads["II"], leads["III"]) / 2

    return leads

def extract_feats(group, leads_to_load):
    leads = OrderedDict(zip(group['LeadID'], group['WaveFormData']))
    leads = augment_leads(leads)

    if set(leads.keys()) == LEAD_ORDER_SET:
        # Take in the correct order if all available (no need to re-order)
        feats = np.stack([leads[key] for key in leads_to_load.index])
        sig_name = LEAD_ORDER
    else:
        # Missing leads - take in whatever order they appear (requires re-order)
        feats = np.stack(list(leads.values()))
        sig_name = np.array(list(leads.keys()))

    feats, avail_leads = reorder_leads(feats, sig_name, leads_to_load)

    fields = {}
    fields['feats'] = feats
    fields['avail_leads'] = str(avail_leads)

    return pd.Series(fields)

SOURCE = "uhn"

def postprocess_muse(meta):
    # Replace 'UNKNOWN' strings with NaN values
    string_cols = meta.columns[meta.apply(is_string_dtype)]
    cols_with_unknowns = (meta[string_cols] == 'UNKNOWN').any()
    cols_with_unknowns = cols_with_unknowns.index[cols_with_unknowns]
    meta.loc[:, cols_with_unknowns] = meta[cols_with_unknowns].replace('UNKNOWN', np.nan)

    meta["age"] = meta["PatientDemographics_PatientAge"].astype(float) * \
        meta["PatientDemographics_AgeUnits"].map({
            "YEARS": 1,
            "MONTHS": 1 / 12,
            "DAYS": 1 / 365
        })

    meta['patient_id'] = meta['PatientDemographics_PatientID'].replace(
        "xxxxx",
        np.nan,
    ).astype(float)

    meta["acq_dt"] = pd.to_datetime(meta["TestDemographics_AcquisitionDate"] + " " + meta["TestDemographics_AcquisitionTime"])
    meta["edit_dt"] = pd.to_datetime(meta["TestDemographics_EditDate"] + " " + meta["TestDemographics_EditTime"])

    meta["dob"] = pd.to_datetime(meta["PatientDemographics_DateofBirth"], format="%m-%d-%Y")

    meta["sex"] = meta['PatientDemographics_Gender'].map({'FEMALE': FEMALE_VALUE, 'MALE': MALE_VALUE})

    return meta

def extract_muse(row, leads_to_load):
    xml_dict = xml_to_dict(row['path'])

    if xml_dict is None:
        logging.info(f"Failed to parse: {row['path']}")
        return {'meta': pd.DataFrame(), 'lead_data': pd.DataFrame(), 'qrs_data': pd.DataFrame()}

    fields = {
        key: pd.Series(xml_dict['RestingECG'][key]).add_prefix(key + '_') \
            for key in MUSE_FIELDS if key in xml_dict['RestingECG']
    }

    # Process the diagnoses
    fields['Diagnosis']['Diagnosis_DiagnosisStatement'] = \
        process_diagnosis(fields['Diagnosis']['Diagnosis_DiagnosisStatement'])

    fields['OriginalDiagnosis']['OriginalDiagnosis_DiagnosisStatement'] = \
        process_diagnosis(fields['OriginalDiagnosis']['OriginalDiagnosis_DiagnosisStatement'])

    # Handle the QRS information
    if 'QRSTimesTypes_QRS' in fields['QRSTimesTypes']:
        qrs_data = fields['QRSTimesTypes']['QRSTimesTypes_QRS']

        # If a single entry, it will be a dictionary
        qrs_data = qrs_data if isinstance(qrs_data, list) else list(qrs_data)
        qrs_data = pd.DataFrame(qrs_data)

        del fields['QRSTimesTypes']['QRSTimesTypes_QRS']
    else:
        qrs_data = pd.DataFrame()

    # Handle the waveforms (keep only the rhythm waveform)
    waveforms = fields['Waveform'].apply(pd.Series)
    waveforms = waveforms[waveforms['WaveformType'] == 'Rhythm'].iloc[0]

    lead_data = pd.Series(waveforms['LeadData']).apply(pd.Series)
    waveforms.drop(index=['LeadData'], inplace=True)
    fields['Waveform'] = waveforms.add_prefix('Waveform_')

    lead_data['WaveFormData'] = lead_data['WaveFormData'].apply(
        lambda data: np.array(array.array("h", base64.b64decode(data)))
    )

    meta = pd.concat([
        extract_feats(lead_data[['LeadID', 'WaveFormData']], leads_to_load),
        pd.concat(fields.values()),
    ])
    meta = pd.concat([
        meta,
        pd.Series(extract_feat_info(meta['feats'], leads_to_load))
    ])

    meta = meta.to_frame().T

    meta['sample_size'] = meta['feats'].apply(lambda feats: feats.shape[1])

    meta.rename({'Waveform_SampleBase': 'sample_rate'}, axis=1, inplace=True)
    meta['sample_rate'] = meta['sample_rate'].astype(int)

    meta_mats = meta[
        ['feats', 'sample_size', 'sample_rate']
    ].copy()
    meta_mats['idx'] = row['idx']
    meta_mats['org_sample_rate'] = meta['sample_rate']
    meta_mats['curr_sample_rate'] = meta['sample_rate']
    meta_mats['org_sample_size'] = meta['sample_size']
    meta_mats['curr_sample_size'] = meta['sample_size']
    meta_mats.drop(['sample_size', 'sample_rate'], axis=1, inplace=True)

    savemat(row['save_path'], meta_mats.iloc[0].to_dict())

    meta.drop('feats', axis=1, inplace=True)
    meta.rename({'path': 'source_path'}, axis=1, inplace=True)

    # Update index as an identifier
    meta = meta.assign(
        new_index=row.name
    ).set_index('new_index').rename_axis(None)
    lead_data = lead_data.assign(
        new_index=row.name
    ).set_index('new_index').rename_axis(None)
    qrs_data = qrs_data.assign(
        new_index=row.name
    ).set_index('new_index').rename_axis(None)

    return {'meta': meta, 'lead_data': lead_data, 'qrs_data': qrs_data}

def main(args):
    records = pd.read_csv(os.path.join(args.processed_root, "records.csv"))
    records["path"] = args.raw_root.rstrip('/') + '/' + records["path"]
    records["save_file"] = filenames_from_paths(records["path"], replacement_ext=".mat")
    records["source"] = SOURCE
    records["dataset"] = SOURCE

    # records = records.iloc[23911:23911 + 5] # No QRs at 153?

    # records = records.iloc[40000: 100000] - Quite possibly
    # records = records.iloc[40000: 70000] - Quite possibly
    # records = records.iloc[40000: 45000] - No
    # records = records.iloc[45000: 50000] - No
    # records = records.iloc[50000: 70000] - Yes
    # records = records.iloc[50000: 60000] - Yes
    # records = records.iloc[50000: 55000] - No
    # records = records.iloc[55000: 60000] - Yes
    # records = records.iloc[59000: 60000] - No

    # records = records.iloc[55000: 60000]
    # records = records.iloc[55000 + 976: 60000] # Single QRS data issue

    # records = records.iloc[60000: 100000] - All good
    # records = records.iloc[100000: 150000] - All good
    # records = records.iloc[150000: 250000] - All good
    # records = records.iloc[250000: 350000] - All good
    # records = records.iloc[350000: 450000] - All good
    # records = records.iloc[450000: 550000] - All good
    # records = records.iloc[550000: 650000] - All good
    # records = records.iloc[650000:]

    pipeline(
        args,
        records,
        SOURCE,
        extract_func=extract_muse,
        postprocess_extraction={'meta': postprocess_muse},
    )

if __name__ == "__main__":
    parser = get_pipeline_parser()
    args = parser.parse_args()
    main(args)
