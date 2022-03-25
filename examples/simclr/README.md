# A Simple Framework for Contrastive Learning of Visual Representations

We provide various perturbation methods with ECG data for SimCLR model, which are listed as:
* Random lead masking
* Powerline noise
* EMG noise
* Baseline shift
* Baseline wander

Before training the model, please follow [these instructions](https://github.com/Jwoo5/fairseq-signals/blob/master/README.md) to install fairseq-signals and prepare required datasets.

# Pre-training a new model
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/total \
    --config-dir examples/simclr/config/pretraining \
    --config-name simclr_rlm
```
If you want to apply more augmentations, refer to `examples/simclr/config/pretraining/simclr_augs.yaml`.

# Fine-tuning a pre-trained model
## Fine-tune on the Cardiac Arrhythmia Classification task
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest/cinc \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir examples/simclr/config/finetuning \
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
    --config-dir examples/simclr/config/finetuning \
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