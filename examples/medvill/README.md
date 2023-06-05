# Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training

This is not an official implementation, but a modified version of [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](https://arxiv.org/abs/2105.11333) to apply to ECG domain, instead of chest X-ray.

Before training the model, plese follow [These instructions](/home/jwoh/ecg/fairseq-signals/README.md) to install fairseq-signals and prepare required datasets.

# Pre-training a new model
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir examples/medvill/config/pretraining \
    --config-name medvill
```

# Fine-tuning a pre-trained model
## Fine-tune on the ECG Question Answering task
We assume the task is formulated as a multi-label classification, and `model.num_labels` is based on the ECG-QA dataset.
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/manifest \
    model.model_path=/path/to/checkpoint.pt \
    model.num_labels=103 \
    --config-dir examples/medvill/config/finetuning/ecg_question_answering \
    --config_name base
```
