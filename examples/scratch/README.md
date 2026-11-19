# Training from scratch
This directory provides configurations to train a model from scratch (i.e., without pre-training) for each downstream task.
Run the following from the root directory of this repository:
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir examples/scratch/$task/$model \
    --config-name $config
```
Fields marked as `???` in a configuration (e.g., `model.num_labels`) are required, so give them in the command (e.g., `model.num_labels=26`) before `--config-dir`. A field that is not written in the configuration should be given with `+` (e.g., `+model.encoder_ffn_embed_dim=3072`).

To fine-tune a pre-trained model instead, refer to the README of each pre-training method (e.g., [W2V+CMSC+RLM](../w2v_cmsc/README.md)).  
To evaluate a trained model, see [Evaluation](../../README.md#evaluation). `infer.yaml` is provided for ECG classification and ECG question answering.

## ECG classification
Configurations in `ecg_classification/`:

| Configuration | Model | Data |
|---|---|---|
| `ecg_transformer/diagnosis_physionet2021` | `ecg_transformer_classifier` | PhysioNet2021 |
| `ecg_transformer/multi_label_cls_path_dataset` | `ecg_transformer_classifier` | ECG-QA converted to classification |
| `ecg_transformer/multi_head_cls_path_dataset` | `ecg_transformer_classifier` | ECG-QA converted to classification |
| `resnet/nejedly2021_total` | `nejedly2021resnet_classifier` | ECG-QA converted to classification |
| `resnet/se_wrn_total` | `se_wrn_classifier` | ECG-QA converted to classification |

* `diagnosis_physionet2021` classifies 26 diagnoses. Prepare the data with `preprocess_physionet2021.py` and `manifest_physionet2021.py` (see [here](../../fairseq_signals/data/ecg/preprocess/README.md)), and set `task.data=/path/to/manifest/finetune`.
* The other configurations are for the upperbound experiments of ECG-QA. Prepare the data with `preprocess_ecgqa_for_classification.py` (see [here](../../fairseq_signals/data/ecg_text/preprocess/README.md#pre-process-for-upperbound-experiments)), which saves the path to each ECG record instead of the signals (`task.path_dataset=true`). Set `model.num_labels` to `83` for the PTB-XL version and `164` for the MIMIC-IV-ECG version.
* `multi_head_cls_path_dataset`, `nejedly2021_total`, and `se_wrn_total` use `multi_head_binary_cross_entropy`, which computes the loss only for the candidate classes of each sample (`target_idx`).

## Patient identification
Configurations in `ecg_identification/`:

| Configuration | Model | Data |
|---|---|---|
| `identification` | `arcface` | PTB-XL, ECG-ID Database, ... |

* Prepare the data manifest with `manifest_identification.py` (see [here](../../fairseq_signals/data/ecg/preprocess/README.md#for-patient-identification-task-ie-ptb-xl)), and set `task.data=/path/to/manifest/identify`.
* Set `model.num_labels` to the number of patients in `train.tsv`. For example, if the last line of `train.tsv` is like `*.mat 2500 69977`, set `model.num_labels=69978`.
* For validation, each sample in `valid_probe.tsv` is identified as the patient of the most similar sample in `valid_gallery.tsv`.
* The outputs of ECG-ID Database contain only lead I, so add `'task.leads_to_load="[0]"'` (see [here](../../fairseq_signals/data/ecg/preprocess/README.md#ecg-id-database)).

## ECG question answering
Configurations in `ecg_question_answering/`:

| Configuration | Model | Inputs |
|---|---|---|
| `ecg_transformer/base`, `ecg_transformer/small` | `ecg_language_transformer_classifier` | ECG and question |
| `resnet_lstm/deeper` | `resnet_lstm_classifier` | ECG and question |
| `blind_transformer/base` | `blind_transformer_qa` | question only |
| `deaf_transformer/base` | `deaf_transformer_qa` | ECG only |

* Prepare the data with `preprocess_ecgqa.py`, and set `model.num_labels` to `103` for the PTB-XL version and `187` for the MIMIC-IV-ECG version (see [here](../../fairseq_signals/data/ecg_text/preprocess/README.md#run-qa-experiments)).
* Except for `deaf_transformer`, the models initialize their word embeddings with those of `bert-base-uncased`, which is downloaded from Hugging Face at the first run. To train them from scratch, add `+model.load_bert_embedding=false`.
* `ecg_transformer/base` and `deaf_transformer/base` are set to use 4 GPUs (`distributed_training.distributed_world_size: 4`). To use a different number of GPUs, change `distributed_training.distributed_world_size` and adjust `optimization.update_freq` to keep the effective batch size.

## ECG segmentation
Configurations in `ecg_segmentation/`:

| Configuration | Model | Data |
|---|---|---|
| `unet/unet3plus_12leads_aug` | `unet_segmentor` | LUDB (12 leads) |
| `unet/unet3plus_aug` | `unet_segmentor` | LUDB (`--separate-leads`) |
| `unet/unet3plus_cgm` | `unet_cgm_segmentor` | LUDB (`--separate-leads`) |
| `unet/unet3plus_aug_cgm` | `unet_cgm_segmentor` | LUDB (`--separate-leads`) |

* Prepare the data with `preprocess_ludb.py` (see [here](../../fairseq_signals/data/ecg/preprocess/README.md#ludb)) and `manifest.py`. `unet3plus_12leads_aug` takes 12-lead inputs, while the others take single-lead inputs, so pre-process LUDB with `--separate-leads` for them. In this case, add `--group-by _` to `manifest.py` to keep all leads of the same record in the same split.
* `*_aug` configurations apply random noise (powerline noise, EMG noise, baseline shift, and baseline wander) to the training samples.
* `*_cgm` configurations also train the classification-guided module (CGM), which classifies the diagnoses of each record. Set `model.num_labels=82`. The loss and metrics of segmentation and classification are logged with the prefixes `0_` and `1_`, respectively.
* To get the outputs of a trained model, see [here](../../README.md#ecg-segmentation).
