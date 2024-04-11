# CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients

Before training the model, please follow [these instructions](https://github.com/Jwoo5/fairseq-signals/blob/master/README.md) to install fairseq-signals and prepare required datasets.

# Prepare training data manifest
Before training, you should prepare training data manifest required for training CMSC model.
```shell script
$ python /path/to/fairseq_signals/data/ecg/preprocess/convert_to_cmsc_manifest.py \
    /path/to/manifest \
    --dest /path/to/converted/manifest

```
The expected results are like:
```
/path/to/converted/manifest
└─ cmsc
    ├─ train.tsv
    ├─ valid.tsv
    └─ test.tsv
```

# Pre-training a new ECG Transformer model
## Pre-train CMSC model
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/cmsc \
    --config-dir examples/cmsc/config/pretraining/ecg_transformer \
    --config-name cmsc
```

# Fine-tuning a pre-trained ECG Transformer model

## Fine-tune on the Cardiac Arrhythmia Classification task
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/finetune \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir examples/cmsc/config/finetuning/ecg_transformer \
    --config-name diagnosis
```
If you want to use CinC score as an evaluation metric, add command line parameters (before `--config-dir`)
`criterion.report_cinc_score=True criterion.weights_file=/path/to/weights.csv`

Note that you can download `weights.csv` file from [here](https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv).

## Fine-tune on the Patient Identification task
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/identify \
    model.model_path=/path/to/checkpoint.pt \
    model.num_labels=$N \
    --config-dir examples/cmsc/config/finetuning/ecg_transformer \
    --config-name identification
```
`$N` should be set to the number of unique patients in the training dataset. You can manually open `/path/to/manifest/identify/train.tsv` file and check the last line of that file. For example, if the last line is like `*.mat 2500 69977`, then `$N` should be set to `69978`.

Note that if you want to train with **PhysioNet2021** dataset and test with **PTB-XL** dataset, prepare data manifest for **PhysioNet2021** with `$valid=0` and **PTB-XL** with `$valid=1.0` seperately and place them to the same manifest directory like this:
```
path/to/manifest/identify
├─ train.tsv
├─ valid_gallery.tsv
└─ valid_probe.tsv
```
Note: `valid_*.tsv` should have been from **PTB-XL** dataset while `train.tsv` should have been from **PhysioNet2021** dataset.