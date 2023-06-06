# Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training

This is not an official implementation, but a modified version of [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](https://arxiv.org/abs/2209.07098) to apply to ECG domain, instead of chest X-ray.

Before training the model, plese follow [these instructions](../../README.md) to install fairseq-signals and prepare required datasets.

# Pre-training a new model
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest \
    model.pretrained_model_path=/path/to/checkpoint.pt \
    --config-dir examples/m3ae/config/pretraining \
    --config-name w2v-cmsc_bert
```
Note that this model requires a pre-trained ECG encoder, provided by `model.pretrained_model_path`.
To pre-train ECG encoder, follow other instructions such as [Wav2Vec 2.0](../../examples/wav2vec2/README.md), [W2V+CMSC+RLM](../../examples/w2v_cmsc/README.md), or any other SSL implementations.

# Fine-tuning a pre-trained model
## Fine-tune on the ECG Question Answering task
We assume the task is formulated as a multi-label classification, and `model.num_labels` is based on the ECG-QA dataset.
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest \
    model.model_path=/path/to/checkpoint.pt \
    model.num_labels=103 \
    --config-dir examples/m3ae/config/finetuning/ecg_question_answering \
    --config_name base
```
