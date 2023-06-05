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
   --dest /path/to/output \
   --meda-dir ./
```
Note that you should provide a path to directory (--meta-dir) containing `ptbxl_database_translated.csv` which has an additional column `report_en` that retains English-translated reports.
If you need to exclude specific samples to be processed, use an additional argument `--exclude`, which is described in `preprocess_ptbxl.py`.

It will output .mat files to `/path/to/output` directory where each sample consists of a 10-second ECG signal and its corresponding report.

## Prepare data manifest
Run:
```shell script
$ python manifest.py \
    /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $valid
```
The expected results are:
```
/path/to/manifest
├─ test.tsv
├─ train.tsv
└─ valid.tsv
```

# ECG-QA
You can download ECG-QA dataset from [here](https://github.com/Jwoo5/ecg-qa).

## Pre-process for QA experiments
Before pre-processing the data, the file structure should be like this:
```
/path/to/ecgqa
├── ...
├── paraphrased
│    ├─ test.json
│    ├─ train.json
│    └─ valid.json
└── template
     ├─ test.json
     ├─ train.json
     └─ valid.json
```
Then, tun:
```shell script
$ python preprocess_ecgqa.py \
    /path/to/ecgqa \
    --ptbxl-data-dir /path/to/ptbxl \
    --dest /path/to/output \
    --apply_paraphrase
```
Note that if you run with `--apply_paraphrase`, the scripts will process the paraphrased version of ECG-QA dataset. Otherwise, it will process the template version.

It will output .mat files to `/path/to/output/$split` directory for each split, and prepare manifest files to `/path/to/output/$split.tsv`.
```
/path/to/output
├── paraphrased
│    ├─ test.tsv
│    ├─ train.tsv
│    ├─ valid.tsv
│    ├─ test
│    │   ├─ 0.mat
│    │   ├─ ...
│    │   └─ 81529.mat
│    ├─ train
│    │   ├─ 0.mat
│    │   ├─ ...
│    │   └─ 265811.mat
│    └─ valid
│        ├─ 0.mat
│        ├─ ...
│        └─ 63776.mat
└── template
     ├─ test.tsv
     ├─ train.tsv
     ├─ valid.tsv
     ├─ test
     │   ├─ 0.mat
     │   ├─ ...
     │   └─ 81529.mat
     ├─ train
     │   ├─ 0.mat
     │   ├─ ...
     │   └─ 265811.mat
     └─ valid
         ├─ 0.mat
         ├─ ...
         └─ 63776.mat
```

## Pre-process for Upperbound experiments
