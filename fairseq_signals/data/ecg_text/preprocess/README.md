# Multi-modal pre-training with ECGs and languages (reports)

## PTB-XL
You can download PTB-XL dataset from [here](https://physionet.org/content/ptb-xl/1.0.3/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

### Pre-process
Before pre-processing the data, the file structure should be like this:
```
/path/to/ptbxl
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
$ python preprocess_ptbxl.py /path/to/ptbxl \
   --dest /path/to/output \
   --meda-dir ./
```
If you need to exclude specific samples to be processed, use an additional argument `--exclude`, which is described in `preprocess_ptbxl.py`.

It will output .mat files to `/path/to/output` directory where each sample consists of a 10-second ECG signal and its corresponding report.

### Prepare data manifest
Run:
```shell script
$ python manifest.py /path/to/data \
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

## MIMIC-IV-ECG
You can download MIMIC-IV-ECG dataset from [here](https://physionet.org/content/mimic-iv-ecg/1.0/) or directly using your terminal:
```shell script
$ wget -r -N -c -np https://physionet.org/files/mimic-iv-ecg/1.0/
```

### Pre-process
Before pre-processing the data, the file structure should be organized like this:
```
/path/to/mimic-iv-ecg
├─ files
│  ├─ p1000
│  │  └─ ...
│  ├─ p1001
│  │  └─ ...
│  └─ ...
├─ machine_measurements.csv
├─ record_list.csv
└─ ...
```
Then, run:
```shell script
$ python preprocess_mimic_iv_ecg.py /path/to/mimic-iv-ecg \
   --dest /path/to/output \
```
If you need to exclude specific samples to be processed, use an additional argument `--exclude`, which is described in `preprocess_mimic_iv_ecg.py`.

It will output .mat files to `/path/to/output` directory where each sample consists of a 10-second ECG signal and its corresponding report.

### Prepare data manifest
Similar with the above (PTB-XL), run:
```shell script
$ python manifest.py /path/to/data \
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

# ECG Question Answering

## ECG-QA
You can download ECG-QA dataset from [here](https://github.com/Jwoo5/ecg-qa).  

## Pre-process for QA experiments
Before pre-processing the data, the file structure should be organized like this:
```
ecgqa
├── ptbxl
│   ├── answers_for_each_template.csv
│   ├── answers.csv
│   ├── test_ecgs.tsv
│   ├── train_ecgs.tsv
│   ├── valid_ecgs.tsv
│   ├── paraphrased
│   │   ├─ test
│   │   │   ├─ 00000.json
│   │   │   │  ...
│   │   │   └─ 80000.json
│   │   ├─ train
│   │   │   ├─ 00000.json
│   │   │   │  ...
│   │   │   └─ 260000.json
│   │   └─ valid
│   │       ├─ 00000.json
│   │       │  ...
│   │       └─ 60000.json
│   └── template
│       ├─ test
│       │   ├─ 00000.json
│       │   │  ...
│       │   └─ 80000.json
│       ├─ train
│       │   ├─ 00000.json
│       │   │  ...
│       │   └─ 260000.json
│       └─ valid
│           ├─ 00000.json
│           │  ...
│           └─ 60000.json
└── mimic-iv-ecg
    ├── ...
    └── (similar with the above)
```
First, you should run this script (existed in the [ECG-QA repository](https://github.com/Jwoo5/ecg-qa)) to map `ecg_id` to the corresponding ECG file path for each ECG-QA sample. For the detailed information about this script, please refer to the dataset repository.
* For PTB-XL version:
    ```shell script
    $ python mapping_ptbxl_samples.py ecgqa/ptbxl \
        --ptbxl-data-dir $ptbxl_dir \
        --dest $dest_dir
    ```
* For MIMIC-IV-ECG version:
    ```shell script
    $ python mapping_mimic_iv_ecg_samples.py ecgqa/mimic-iv-ecg \
        --mimic-iv-ecg-data-dir $mimic_iv_ecg_dir \
        --dest $dest_dir
    ```

Then, run:
```shell script
$ python preprocess_ecgqa.py /path/to/ecgqa \
    --dest /path/to/output \
    --apply_paraphrase
```
\*`/path/to/ecgqa` should be consistent with `$dest_dir` in the mapping script (i.e., `mapping_ptbxl_samples.py` or `mapping_mimic_iv_ecg_samples.py`).  
Note that if you run with `--apply_paraphrase`, the scripts will process the paraphrased version of ECG-QA dataset. Otherwise, it will process the template version.

It will output .mat files to `/path/to/output/template/$split` or `/path/to/output/paraphrased/$split` directory for each split, and prepare manifest files as `/path/to/output/template/$split.tsv` or `/path/to/output/paraphrased/$split.tsv` (see below).
```
/path/to/output
├── paraphrased
│    ├─ test.tsv
│    ├─ train.tsv
│    ├─ valid.tsv
│    ├─ test
│    │   ├─ 0.mat
│    │   └─ ...
│    ├─ train
│    │   ├─ 0.mat
│    │   └─ ...
│    └─ valid
│        ├─ 0.mat
│        └─ ...
└── template
     ├─ test.tsv
     ├─ train.tsv
     ├─ valid.tsv
     ├─ test
     │   ├─ 0.mat
     │   └─ ...
     ├─ train
     │   ├─ 0.mat
     │   └─ ...
     └─ valid
         ├─ 0.mat
         └─ ...
```

## Run QA experiments
Please follow the instructions in the [root directory](../../../../) to install `fairseq-signals` before running experiments.

Run:
```shell script
$ fairseq-hydra-train task.data=/path/to/output/paraphrased \
    model.num_labels=$num_labels \
    --config-dir /fairseq-signals/examples/scratch/ecg_question_answering/$model_name \
    --config-name $model_config_name
```
$num_labels: the number of answers specified in `answers.csv`. In other words, `103` for ptb-xl version, and `187` for mimic-iv-ecg version (Note that the answer `none` is not counted because it is regarded as an "empty label").  
$model_name: the name of the ECG-QA model (e.g., `ecg_transformer`)  
$model_config_name the name of the configuration file (e.g., `base`)

If you want to fine-tune a pre-trained model on QA task, please refer to detailed README for each pre-training model implementation.
* [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](../../../../examples/m3ae/README.md)
* [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](../../../../examples/medvill/README.md)

## Pre-process for Upperbound experiments
For detailed description of upperbound experiments, please refer to the original paper.  
To convert QA samples into ECG classification format, run:
```shell script
$ python preprocess_ecgqa_for_classification.py /path/to/ecgqa \
    --dest /path/to/output \
```
\*Note that before running this script, you should have mapped ecg_id to the corresponding ECG file path by `mapping_ptbxl_samples.py` or `mapping_mimic_iv_ecg_samples.py`.  

Similar to other preprocessing scripts, it will output .mat files to `/path/to/output/$split` directory for each split, and prepare manifest files to `/path/to/output/$split.tsv`.  
`grounding_class.csv` provides the class information for the encoded class indices.
```
/path/to/output
├─ grounding_class.csv
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
    model.num_labels=$num_labels \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir /fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer/grounding_classification \
    --config-name base_total
```
$num_labels: the number of attributes for the upperbound experiments. `83` for ptb-xl version, and `164` for mimic-iv-ecg version (see `grounding_class.csv`).  
Note that you need to pass the path to the pretrained model checkpoint through `model.model_path`.  
To pre-train the model, refer to [here](../../../../examples/w2v_cmsc/README.md).  

For Resnet + Attention model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=$num_labels \
    --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
    --config-name nejedly2021_total
```
$num_labels: the number of attributes for the upperbound experiments. `83` for ptb-xl version, and `164` for mimic-iv-ecg version (see `grounding_class.csv`).  

For SE-WRN model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=$num_labels \
    --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
    --config-name se_wrn_total
```
$num_labels: the number of attributes for the upperbound experiments. `83` for ptb-xl version, and `164` for mimic-iv-ecg version (see `grounding_class.csv`).  