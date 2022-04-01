# PhysioNet2021
You can download PhysioNet2021 datasets at the Data Access section from [here](https://moody-challenge.physionet.org/2021/) or directly using your terminal:
```shell script
$ wget -O WFDB_CPSC2018.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/
wget -O WFDB_CPSC2018_2.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/
wget -O WFDB_StPetersburg.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining//WFDB_StPetersburg.tar.gz/
wget -O WFDB_PTB.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/
wget -O WFDB_PTBXL.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/
wget -O WFDB_Ga.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/
wget -O WFDB_ChapmanShaoxing.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/
wget -O WFDB_Ningbo.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ningbo.tar.gz/
```
We recommend you to download those datasets excluding **StPetersburg** and **PTB** for better reproducing since they contain too long ECG records or have differrent sampling rates with others.

## Pre-process
Before pre-processing the data, the file structure should be like this:
```
path/to/WFDB
├─ WFDB_ChapmanShaoxing
│  ├─ *.hea
│  └─ *.mat
├─ WFDB_CPSC2018
│  ├─ *.hea
│  └─ *.mat
├─ WFDB_CPSC2018_2
│  ├─ *.hea
│  └─ *.mat
├─ ...
└─ WFDB_PTBXL
   ├─ *.hea
   └─ *.mat
```
Then, run:
```shell script
$ python preprocess_physionet2021.py \
   /path/to/WFDB/ \
   --dest /path/to/output \
   --workers $N
```
$N is the number of workers for multi-processing.

It will output .mat files to `/path/to/output` directory after encoding labels (age, diagnosis, patient id), and repeatedly cropping 5 seconds for each ECG record.

# PTB-XL
You can download PTB-XL dataset from [here](https://physionet.org/content/ptb-xl/1.0.1/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/
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
   /path/to/ptbxl/records500/ \
   --dest /path/to/output
```
It will filter ECG samples to have at least two corresponding sessions according to `patient id` and randomly crop 5 seconds.

# Prepare data manifest
Run:
```shell script
$ python manifest.py \
   /path/to/data/ \
   --dest /path/to/manifest \
   --valid-percent $valid
```
The expected results are like:
```
/path/to/manifest
├─ total
│  └─ train.tsv
└─ cinc
   ├─ train.tsv
   ├─ valid.tsv
   └─ test.tsv
```
Note that `total/train.tsv` is used for pre-training, and `cinc/*.tsv` is used for fine-tuning with cardiac arrhythmia classification task.

For patient identification task, run:
```shell script
$ python manifest_identification.py \
   /path/to/data \
   --dest /path/to/manifest \
   --valid-percent $valid
```
The expected results are like:
```shell script
/path/to/manifest
└─ identify
   ├─ train.tsv
   ├─ valid_gallery.tsv
   └─ valid_probe.tsv
```
$valid should be set to percentage of training data to use for validation.


If you want to combine many datasets to compose train, valid, and test splits, we recommend you to manifest them separately and combine them manually. For example, if you want to train a patient identification model using **PhysioNet2021** and test with **PTB-XL**, prepare data manifest for **PhysioNet2021** with `$valid=0` and **PTB-XL** with `$valid=1.0` seperately and place them to the same manifest directory like this:
```shell script
path/to/manifest/identify
├─ train.tsv
├─ valid_gallery.tsv
└─ valid_probe.tsv
```
Note: `valid_gallery.tsv` and `valid_probe.tsv` should have been from **PTB-XL** dataset.