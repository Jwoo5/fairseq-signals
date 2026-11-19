# PhysioNet2021
You can download PhysioNet2021 datasets at the Data Access section from [here](https://physionet.org/content/challenge-2021/1.0.3/#files) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/
```

## Pre-process
Before pre-processing the data, the file structure should be organized like this:
```
path/to/.../1.0.3/training
├─ chapman_shaoxing
│  ├─ g1
│  │  ├─ *.hea
│  │  └─ *.mat
│  ├─ ...
│  └─ g11
│     ├─ *.hea
│     └─ *.mat
├─ cpsc_2018
│  ├─ g1
│  │  ├─ *.hea
│  │  └─ *.mat
│  ├─ ...
│  └─ g7
│     ├─ *.hea
│     └─ *.mat
├─ ...
└─ ptb-xl
   ├─ g1
   │  ├─ *.hea
   │  └─ *.mat
   ├─ ...
   └─ g22
      ├─ *.hea
      └─ *.mat
```
Then, run:
```shell script
$ python preprocess_physionet2021.py \
   /path/to/WFDB/ \
   --meta-dir /path/to/meta \
   --dest /path/to/output \
   --workers $N
```
$N is the number of workers for multi-processing.  
`--meta-dir` should point to the directory containing `weights.csv`, which is used for labeling diagnoses. It is not included in the PhysioNet2021 dataset, so download it from [here](https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv). If `--meta-dir` is omitted, the script looks for `weights.csv` in `/path/to/WFDB/`.

It will output .mat files to `/path/to/output` directory after encoding labels (age, diagnosis, patient id), and repeatedly cropping 5 seconds for each ECG record.  
Note that this script does not process `ptb` and `st_petersburg_incart` by default since these two contain too long ECG records or have different sampling rates with others.

# PTB-XL
You can download PTB-XL dataset from [here](https://physionet.org/content/ptb-xl/1.0.3/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

## Pre-process
Before pre-processing the data, the file structure should be like this:
```
path/to/ptbxl
├─ records100
│  └─ ...
├─ records500
│  ├─ 00000
│  ├─ 01000
│  ├─ ...
│  └─ 21000
│    ├─ *.dat
│    └─ *.hea
└─ ...
```
Then, run:
```shell script
$ python preprocess_ptbxl.py \
   /path/to/ptbxl \
   --dest /path/to/output
```
Note that the root directory should be `/path/to/ptbxl` (which contains `ptbxl_database.csv`), not `/path/to/ptbxl/records500`.  
It will filter ECG samples to have at least two corresponding sessions according to `patient id` and randomly crop 5 seconds.  
To use only normal ECGs (records with `NORM` in `scp_codes`), add `--only-norm`. In this case, the two-session filter is applied after excluding the other records.  
Note that the outputs contain patient ids but no diagnostic labels, so they are meant for the patient identification task (see [below](#for-patient-identification-task-ie-ptb-xl)).

# LUDB
You can download LUDB dataset from [here](https://physionet.org/content/ludb/1.0.1/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/
```

## Pre-process
Given the `data` directory of LUDB (which contains `*.dat`, `*.hea`, and the annotation files for each lead such as `*.i`, `*.ii`, ..., `*.v6`), run:
```shell script
$ python preprocess_ludb.py \
   /path/to/ludb/data \
   --dest /path/to/output
```
Note that the given directory is not searched recursively, so it should be `/path/to/ludb/data`, not `/path/to/ludb`.  
It will output `{record}.mat` for each record, which contains:
* `feats`: 12-lead ECG signals
* `segment_label`: the class of each time step (`0`: P wave, `1`: QRS complex, `2`: T wave, `3`: none), where the boundaries of each wave annotated for each lead are aligned across the leads
* `segment_mask`: `True` for the time steps out of the annotated range (from the first to the last annotated wave), which are excluded from the loss and the metrics
* `label`: multi-hot labels for the 82 diagnoses written in the headers, which are used only by the segmentation models with the classification-guided module (CGM, `task.apply_cgm=true`)

To save each lead separately with its own annotations (e.g., for single-lead segmentation models), add `--separate-leads`. Then, it will output `{record}_{lead}.mat` for each lead of each record.  
Records that are not sampled at `--sample-rate` (default: `500`) are skipped, unless `--resample` is given to resample them to `--sample-rate`.  
See [here](../../../../examples/scratch/README.md#ecg-segmentation) for the configurations to train segmentation models with these outputs.

Note: this script reads the annotations with `wfdb.rdann`, which raises `OverflowError` with NumPy 2 if wfdb < 4.2.0 is installed. In this case, upgrade wfdb (`pip install -U wfdb`).

# ECG-ID Database
You can download ECG-ID Database from [here](https://physionet.org/content/ecgiddb/1.0.0/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/ecgiddb/1.0.0/
```

## Pre-process
Given the root directory of ECG-ID Database (which contains `Person_01`, `Person_02`, ...), run:
```shell script
$ python preprocess_ecgiddb.py \
   /path/to/ecgiddb \
   --dest /path/to/output
```
It will take the filtered lead I signal of each record, randomly crop `--sec` seconds (default: `5`), and output `{patient_id}_{record}.mat`, where an integer patient id is assigned to each `Person_*` directory.  
Then, prepare data manifest for the patient identification task as described [below](#for-patient-identification-task-ie-ptb-xl).  
Note that the outputs contain only a single lead (lead I). To train a model for 12-lead ECGs (e.g., [`identification.yaml`](../../../../examples/scratch/ecg_identification/identification.yaml)), add `'task.leads_to_load="[0]"'` to the training command so that lead I is placed at the first lead and the other leads are padded with zeros (`task.enable_padding_leads=true`).

Note: as with LUDB, this script reads the annotations with `wfdb.rdann`, which requires wfdb >= 4.2.0 with NumPy 2.

# Prepare data manifest
### For pretraining and finetuning with PhysioNet2021

Run:
```shell script
$ python manifest_physionet2021.py \
   /path/to/data/ \
   --dest /path/to/manifest \
   --finetune-valid-percent $valid
```
`--finetune-valid-percent` (default: `0.1`) is the percentage of the fine-tuning data to use as the validation and test sets, which are excluded from the pre-training data. You can also set `--pretrain-valid-percent` (default: `0.0`) for the pre-training split, and choose which subsets are used for each stage with `--pretrain_subset` and `--finetune_subset`.
To keep all crops of the same ECG record (saved as `{record}_{i}.mat` by `preprocess_physionet2021.py`) in the same split, add `--group-by _`. It puts files into the same split if their paths are identical up to the last occurrence of the given delimiter. Note that this groups crops by record, not by patient.

The expected results are like:
```
/path/to/manifest
├─ pretrain
│  ├─ train.tsv
│  ├─ valid.tsv
│  └─ test.tsv
└─ finetune
   ├─ train.tsv
   ├─ valid.tsv
   └─ test.tsv
```
Note that `pretrain/train.tsv` is used for pre-training, and `finetune/*.tsv` is used for fine-tuning such as cardiac arrhythmia classification task. `pretrain/valid.tsv` and `pretrain/test.tsv` are always created, but they contain no samples unless `--pretrain-valid-percent` is greater than 0.

### For patient identification task (i.e., PTB-XL)
This also applies to the outputs of ECG-ID Database.

Run:
```shell script
$ python manifest_identification.py \
   /path/to/data \
   --dest /path/to/manifest \
   --valid-percent $valid
```
The expected results are like:
```
/path/to/manifest
└─ identify
   ├─ train.tsv
   ├─ valid_gallery.tsv
   └─ valid_probe.tsv
```
$valid should be set to percentage of training data to use for validation.


If you want to combine many datasets to compose train, valid, and test splits, we recommend you to manifest them separately and combine them manually. For example, if you want to train a patient identification model using **PhysioNet2021** and test with **PTB-XL**, prepare data manifest for **PhysioNet2021** with `$valid=0` and **PTB-XL** with `$valid=1.0` seperately and place them to the same manifest directory like this:
```
path/to/manifest/identify
├─ train.tsv
├─ valid_gallery.tsv
└─ valid_probe.tsv
```
Note: `valid_gallery.tsv` and `valid_probe.tsv` should have been from **PTB-XL** dataset.

### Otherwise (i.e., LUDB, ...)
Run:
```shell script
$ python manifest.py /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $valid
```
`$valid` is applied to each of the validation and test sets (e.g., `0.1` results in a 80/10/10 split). Similar to `manifest_physionet2021.py`, you can use `--group-by` to put files sharing the same prefix (up to the last occurrence of the given delimiter) into the same split.  
For example, if LUDB is pre-processed with `--separate-leads` (`{record}_{lead}.mat`), add `--group-by _` to keep all leads of the same record in the same split.

The expected results are:
```
/path/to/manifest
├─ test.tsv
├─ train.tsv
└─ valid.tsv
```