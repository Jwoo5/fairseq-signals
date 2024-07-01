This document outlines `fairseq-signals`' flexible, end-to-end, multi-source data preprocessing pipeline. It is complete with: record extraction; waveform extraction, preprocessing, and analysis; defining data splits (e.g., train, valid, test); and compiling multiple data sources into one manifest (which is very useful for pretraining).

# Set up
For simplicity, we recommended defining a root directory in which all processed data sources will be stored, where each source becomes its own subdirectory.
```
PROCESSED_ROOT=".../data"

cd .../fairseq-signals/scripts/preprocess/ecg
```

# Signal Processing

The signal processing must be performed one dataset at a time so as to appropriately create unique integer sample IDs and record them in the `manifest_file`. The sample IDs are used to attribute labels to the samples such that they do not need to be stored within the ECG waveform files themselves. This allows for quick label manipulation and exploration.

The `records` scripts grab all relevant file paths, as well as any metadata which might already be stored in a tabular format.

The `signals` scripts extracts file metadata and performs preprocessing, specifically converting the raw waveform into a `.mat` format (`org`), performing operations such as resampling and standardization (`preprocessed`), and finally segmenting the ECGs into non-overlapping portions of a desired length (`segmented`).

The `splits` script partitions the data, for example, into `train`, `valid`, and `test` sets which can be used for experimentation. Multiple different strategies are available for use.


## PhysioNet 2021
Designed for and tested with PhysioNet 2021 v1.0.3.

```
PHYSIONET_ROOT=".../physionet.org/files/challenge-2021/1.0.3/training"

python physionet2021_records.py \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --physionet_root "$PHYSIONET_ROOT"

python physionet2021_signals.py \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --raw_root "$PHYSIONET_ROOT" \
    --manifest_file "$PROCESSED_ROOT/manifest.csv"

python ../splits.py \
    --strategy "random" \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --filter_cols "nan_any,constant_leads_any" \
    --dataset_subset "cpsc_2018, cpsc_2018_extra, georgia, ptb-xl, chapman_shaoxing, ningbo" # Excludes 'ptb' and 'st_petersburg_incart'
```


## MIMIC-IV-ECG
Designed for and tested with MIMIC-IV-ECG v1.0 (and optionally MIMIC-IV v2.2).

```
MIMIC_IV_ECG_ROOT=".../MIMIC-IV-ECG"
MIMIC_IV_ROOT=".../mimiciv/2.2" # Optional

python mimic_iv_ecg_records.py \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --mimic_iv_ecg_root "$MIMIC_IV_ECG_ROOT" \
    --mimic_iv_root "$MIMIC_IV_ROOT" # Optional

python mimic_iv_ecg_signals.py \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --raw_root "$MIMIC_IV_ECG_ROOT" \
    --manifest_file "$PROCESSED_ROOT/manifest.csv"

python ../splits.py \
    --strategy "grouped" \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --group_col "subject_id" \
    --filter_cols "nan_any,constant_leads_any"
```


## MUSE ECGs
There are also scripts for extracting MUSE v8.0.2.10132 XML ECGs, however, the generalizability of this code is unclear at this time.


# Manifest Creation
The `manifests` script compiles the splits of multiple sources into one singular file which can be used for training.

```
MANIFEST_DIR = "$PROCESSED_ROOT/manifests/..."

cd .../fairseq-signals/scripts/preprocess

python manifests.py \
    --split_file_paths "$PROCESSED_ROOT/physionet2021/segmented_split.csv,$PROCESSED_ROOT/mimic_iv_ecg/segmented_split.csv" \
    --save_dir "$MANIFEST_DIR"

# If training a CMSC model
python .../fairseq_signals/data/ecg/preprocess/convert_to_cmsc_manifest.py \
    "$MANIFEST_DIR" \
    --dest "$MANIFEST_DIR"
```