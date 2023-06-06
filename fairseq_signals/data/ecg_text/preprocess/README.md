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

## Run experiments
Please follow the instructions in the [root directory](../../../../) to install `fairseq-signals` before running experiments.

### Pre-training
For multi-modal pre-training of ECGs and reports, please refer to detailed README for each model implementation.
* [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](../../../../examples/m3ae/README.md)
* [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](../../../../examples/medvill/README.md)

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
Then, run:
```shell script
$ python preprocess_ecgqa.py /path/to/ecgqa \
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

## Run QA experiments
Please follow the instructions in the [root directory](../../../../) to install `fairseq-signals` before running experiments.

Run:
```shell script
$ fairseq-hydra-train task.data=/path/to/output/paraphrased \
    model.num_labels=103 \
    --config-dir /fairseq-signals/examples/scratch/ecg_question_answering/$model_name \
    --config-name $model_config_name
```
$model_name: the name of the ECG-QA model (e.g., `ecg_transformer`)  
$model_config_name the name of the configuration file (e.g., `base`)

If you want to fine-tune a pre-trained model on QA task, please refer to detailed README for each pre-training model implementation.
* [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](../../../../examples/m3ae/README.md)
* [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](../../../../examples/medvill/README.md)

## Pre-process for Upperbound experiments
For detailed description of upperbound experiments, refer to original paper.  
To convert QA samples into ECG classification format, run:
```shell script
$ python preprocess_ecgqa_for_classification.py /path/to/ecgqa \
    --ptbxl-data-dir /path/to/ptbxl \
    --dest /path/to/output \
```

Similar to other preprocessing scripts, it will output .mat files to `/path/to/output/$split` directory for each split, and prepare manifest files to `/path/to/output/$split.tsv`.
```
/path/to/output
├─ test.tsv
├─ train.tsv
├─ valid.tsv
├─ test
│   ├─ 1_0_1_2_3_4_5.mat
│   ├─ ...
│   └─ 21837_entire.mat
├─ train
│   ├─ 3_0_4_10_11.mat
│   ├─ ...
│   └─ 21835_entire.mat
└─ valid
    ├─ 2_0_1_2_4_5_6_7_8_10_11.mat
    ├─ ...
    └─ 21816_entire.mat
```

## Run upperbound experiments
For W2V+CMSC+RLM:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir /fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer/grounding_classification \
    --config-name base_total
```
Note that you need to pass the path to the pretrained model checkpoint through `model.model_path`.  
To pre-train the model, refer to [here](../../../../examples/w2v_cmsc/README.md).

For Resnet + Attention model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
    --config-name nejedly2021_total
```

For SE-WRN model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
    --config-name se_wrn_total
```