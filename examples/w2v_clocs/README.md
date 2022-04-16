# Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram

This is the official pytorch code implementation of [Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram](https://arxiv.org/abs/2203.06889).

Before training the model, please follow [these instructions](https://github.com/Jwoo5/fairseq-signals/blob/master/README.md) to install fairseq-signals and prepare required datasets.

# Prepare training data manifest
Before training, you should prepare training data manifest required for training CLOCS model.
```shell script
$ python /path/to/fairseq_signals/data/ecg/preprocess/convert_to_clocs_manifest.py \
    /path/to/total/train.tsv \
    --dest /path/to/manifest
```
The expected results are like:
```
/path/to/manifest
├─ cmsc
│  └─ train.tsv
├─ cmlc
│  └─ train.tsv
└─ cmsmlc
   └─ train.tsv
```

# Pre-training a new model
This configuration was used for the `W2V+CMSC+RLM` model pre-trained on the `PhysioNet2021` dataset in the original paper.

```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/cmsc \
    --config-dir examples/w2v_clocs/config/pretraining \
    --config-name w2v_cmsc_rlm
```

# Fine-tuning a pre-trained model

## Fine-tune on the Cardiac Arrhythmia Classification task
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/cinc \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir examples/w2v_clocs/config/finetuning \
    --config-name diagnosis
```
If you want to use CinC score as an evaluation metric, add command line parameters (before `--config-dir`)
`criterion.report_cinc_score=True criterion.weights_file=/path/to/weights.csv`

Note that you can download `weights.csv` file from [here](https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv).

## Fine-tune on the Patient Identification task
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/identify \
    task.num_labels=$N \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir examples/w2v_clocs/config/finetuning \
    --config-name identification
```
`$N` should be set to the number of unique patients in the training dataset. You can manually open `/path/to/manifest/identify/train.tsv` file and check the last line of that file. For example, if the last line is like `*.mat 2500 69977`, then `$N` should be set to `69978`.

Note that if you want to train with **PhysioNet2021** dataset and test with **PTB-XL** dataset, prepare data manifest for **PhysioNet2021** with `$valid=0` and **PTB-XL** with `$valid=1.0` seperately and place them to the same manifest directory like this:
```shell script
path/to/manifest/identify
├─ train.tsv
├─ valid_gallery.tsv
└─ valid_probe.tsv
```
Note: `valid_*.tsv` should have been from **PTB-XL** dataset while `train.tsv` should have been from **PhysioNet2021** dataset.

# Citation
Please cite as:
```
@inproceedings{oh2022lead,
  title={Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram},
  author={Oh, Jungwoo and Chung, Hyunseung and Kwon, Joon-myoung and Hong, Dong-gyun and Choi, Edward},
  booktitle={Conference on Health, Inference, and Learning},
  pages={338--353},
  year={2022},
  organization={PMLR}
}
```