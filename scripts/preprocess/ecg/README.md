This document outlines `fairseq-signals`' flexible, end-to-end, multi-source data preprocessing pipeline. It is complete with: record extraction; waveform extraction, preprocessing, and analysis; defining data splits (e.g., train, valid, test); and compiling multiple data sources into one manifest (which is very useful for pretraining).

# Overview

**The signal processing must be performed one dataset at a time** so as to appropriately create unique integer sample IDs and record them in the `manifest_file`. The sample IDs are useful for attributing labels to samples such that they do not need to be stored within the ECG waveform files themselves.

The `*_records.py` scripts grab all relevant file paths, as well as any metadata which might already be stored in a tabular format (`records.csv`). This is helpful for compiling information available without loading any waveforms, as well as providing an opportunity to filter out select samples before processsing.

The `*_signals.py` scripts extracts file metadata (`meta.csv`) and performs preprocessing, specifically converting the raw waveform into a `.mat` format (`org/`), performing operations such as resampling and standardization (`preprocessed/`), and finally segmenting the ECGs into non-overlapping portions of a desired length (`segmented/`).

The `splits.py` script partitions the data, for example, into `train`, `valid`, and `test` sets which can be used for experimentation (`meta_split.csv`, `segmented_split.csv`). Multiple different strategies are available for use under this same script.

Here is an example of a resulting directory structure after running the scripts for PhysioNet 2021 and MIMIC-IV-ECG:
```
data
│   manifest.csv 
│
└───physionet2021
│   │   meta.csv
│   │   meta_split.csv
│   │   records.csv
│   │   segmented.csv
│   │   segmented_split.csv
│   │
│   └───org
│   │   │   ...
│   │
│   └───preprocessed
│   |   │   ...
│   │
│   └───segmented
│   |   │   ...
│   │
└───mimic_iv_ecg
│   │   meta.csv
│   │   meta_split.csv
│   │   records.csv
│   │   segmented.csv
│   │   segmented_split.csv
│   │
│   └───org
│   │   │   ...
│   │
│   └───preprocessed
│   |   │   ...
│   │
│   └───segmented
│   |   │   ...
```

# Usage
For simplicity, we recommended defining a root directory in which all processed data sources will be stored, where each source becomes its own subdirectory.
```
PROCESSED_ROOT=".../data"

cd .../fairseq-signals/scripts/preprocess/ecg
```

## Datasets
### PhysioNet 2021
Designed for and tested with PhysioNet 2021 v1.0.3. This data source stores its waveforms as WFDB files.

```
PHYSIONET_ROOT=".../physionet.org/files/challenge-2021/1.0.3/training"

python physionet2021_records.py \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --raw_root "$PHYSIONET_ROOT"

python physionet2021_signals.py --help

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

### MIMIC-IV-ECG
Designed for and tested with MIMIC-IV-ECG v1.0 (and optionally MIMIC-IV v2.2). This data source stores its waveforms as WFDB files.

```
MIMIC_IV_ECG_ROOT=".../MIMIC-IV-ECG"
MIMIC_IV_ROOT=".../mimiciv/2.2" # Optional

python mimic_iv_ecg_records.py \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --raw_root "$MIMIC_IV_ECG_ROOT" \
    --mimic_iv_root "$MIMIC_IV_ROOT" # Optional

python mimic_iv_ecg_signals.py --help

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

### CODE-15%
Designed for and tested with CODE-15% Version 1.0.0 (https://zenodo.org/records/4916206). This data source stores its waveforms in an HDF5 file.
```
CODE_15_ROOT=".../code_15"

# There is no need for a code_15_records.py file
# Simply rename exams.csv to records.csv and place it in the processed root

python code_15_signals.py --help

python code_15_signals.py \
    --processed_root "$PROCESSED_ROOT/code_15" \
    --raw_root "$CODE_15_ROOT" \
    --manifest_file "$PROCESSED_ROOT/manifest.csv"
```

### MUSE ECGs
There are also scripts for extracting MUSE v8.0.2.10132 XML ECGs, however, the generalizability of this code is unclear at this time. This data source stores its waveforms as XML files.


## Manifest Creation
The `manifests.py` script converts split files into a directory of e.g., `train.tsv`, `valid.tsv`, `test.tsv`

save directory can be specified in the framework using the `--task.data` argument

 which can be specified to `fairseq-signals`. This is not to be confused with the `manifest.csv`

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

## Binary Classification

Prepares a CSV of binary columns for classification, creating a NumPy array, label definition, and positive weights for positive sample re-weighting.

```
python prepare_clf_labels.py \
    --output_dir "$PROCESSED_ROOT/physionet2021/labels"
    --labels "$PROCESSED_ROOT/physionet2021/labels/labels.csv"
    --splits "$PROCESSED_ROOT/physionet2021/segmented_split.csv"
```

# Curating Datasets

To curate a dataset, new `*_records.py` and `*_signals.py` files must be created. The `example_records.py` and `example_signals.py` files provide useful documentation, outlining the structure to follow when curating a new dataset.

A `*_records.py` file may not be required if a dataset already has a CSV of records, or it may be as simple as loading this file and renaming certain columns. The only requirement is a `path` column in the resulting `records.csv`, which is used by the `extract_func` defined in the corresponding `*_signals.py` file.

A `*_signals.py` file is required, however, there may be code available to help load the new dataset signals. For example, the `extract_wfdb` function used by both the PhysioNet 2021 and MIMIC-IV-ECG datasets.
