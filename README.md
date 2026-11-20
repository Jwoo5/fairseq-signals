# Fairseq-signals

Fairseq-signals is a collection of deep learning models for ECG data processing based on the [`fairseq`](https://github.com/pytorch/fairseq).

We provide implementations of various deep learning methods on ECG data, including official implementations of our works.

### List of implemented papers:
* [Deep learning based ECG segmentation for delineation of diverse arrhythmias](https://arxiv.org/abs/2304.06237)
* [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](https://arxiv.org/abs/2209.07098)
* [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](https://arxiv.org/abs/2105.11333)
* [Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram](https://arxiv.org/abs/2203.06889)*
* [3KG: Contrastive Learning of 12-Lead Electrocardiograms using Physiologically-Inspired Augmentations](https://arxiv.org/abs/2106.04452)
* [CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients](https://arxiv.org/abs/2005.13249)
* [wav2vec 2.0: A Framework for Self-supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [ECG-FM: An Open Electrocardiogram Foundation Model](https://arxiv.org/abs/2408.05178)*

\* denotes for an official implementation

We will keep implementing new methods in this repo. If you have any recommendations, please contact us via an issue or an e-mail.

# Requirements and Installation
* [PyTorch](https://pytorch.org) version >= 1.5.0
* Python version >= 3.7
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
git clone https://github.com/Jwoo5/fairseq-signals
cd fairseq-signals
pip install --editable ./
```

* **To preprocess ECG datasets**: the scripts in `fairseq_signals/data/*/preprocess` only require the packages installed together with fairseq-signals (e.g., `pandas`, `scipy`, `wfdb`). For the multi-source pre-processing pipeline in `scripts/preprocess`, see [here](scripts/preprocess/ecg/README.md#requirements) for additional requirements.
* **To build cython components**: `python setup.py build_ext --inplace`
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* **To apply signal filtering** (`task.filter=true`): `pip install neurokit2`

# Getting Started
## For uni-modal tasks (ECG Classification, ...)
### Prepare ECG dataset
We provide pre-processing codes for various ECG datasets.

* [PhysioNet2021](https://physionet.org/content/challenge-2021/1.0.3/#files)
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (for patient identification)
* [LUDB](https://physionet.org/content/ludb/1.0.1/) (for ECG segmentation)
* [ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/) (for patient identification)

The commands for LUDB and ECG-ID Database are described [here](fairseq_signals/data/ecg/preprocess/README.md).  
We also provide a multi-source pre-processing pipeline in [`scripts/preprocess`](scripts/preprocess/ecg/README.md), which processes PhysioNet2021, MIMIC-IV-ECG, CODE-15%, and MUSE ECGs and compiles them into one manifest (e.g., for pre-training).

### Pre-process
Given a directory that contains WFDB directories to be pre-processed for **PhysioNet2021**:

```shell script
$ python fairseq_signals/data/ecg/preprocess/preprocess_physionet2021.py \
    /path/to/physionet2021/ \
    --meta-dir /path/to/meta \
    --dest /path/to/output \
    --workers $N
```
`--meta-dir` should point to the directory containing `weights.csv`, which is used for labeling. It is not included in the PhysioNet2021 dataset, so download it from [here](https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv). If `--meta-dir` is omitted, the script looks for `weights.csv` in the root directory.

Given the root directory of PTB-XL (which contains `ptbxl_database.csv`):
```shell script
$ python fairseq_signals/data/ecg/preprocess/preprocess_ptbxl.py \
    /path/to/ptbxl \
    --dest /path/to/output
```
Note that this script keeps only patients with at least two ECG records and saves patient ids without diagnostic labels, so its outputs are meant for the patient identification task.

### Prepare data manifest
For **PhysioNet2021**, run the following to prepare manifests for pre-training (`/path/to/manifest/pretrain`) and fine-tuning (`/path/to/manifest/finetune`), which are used in the [examples](#examples):
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest_physionet2021.py \
    /path/to/data/ \
    --dest /path/to/manifest \
    --finetune-valid-percent $valid
```
For patient identification (e.g., **PTB-XL**, **ECG-ID Database**):
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest_identification.py \
    /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $valid
```
For other pre-processed datasets (e.g., **LUDB**):
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest.py \
    /path/to/data/ \
    --dest /path/to/manifest \
    --valid-percent $valid
```
For `manifest_physionet2021.py` and `manifest.py`, `$valid` is applied to each of the validation and test sets (e.g., `0.1` results in a 80/10/10 split).

Both `manifest_physionet2021.py` and `manifest.py` also support `--group-by`, which puts files into the same split if their paths are identical up to the last occurrence of the given delimiter. Since the PhysioNet2021 pre-processing saves multiple crops of one ECG record as `{record}_{i}.mat`, add `--group-by _` to keep all crops of the same record in the same split. Note that this groups crops by record, not by patient.

Please find more details about pre-processing and data manifest from [here](fairseq_signals/data/ecg/preprocess/README.md).

## For multi-modal tasks (Multi-modal pre-training or ECG question answering)
### Prepare ECG dataset
We provide pre-processing codes for the following datasets.
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
* [ECG-QA](https://github.com/Jwoo5/ecg-qa)

### Pre-process
For multi-modal pre-training of ECGs with reports using the **PTB-XL** dataset:
```shell script
$ python fairseq_signals/data/ecg_text/preprocess/preprocess_ptbxl.py \
   /path/to/ptbxl \
   --dest /path/to/output \
```
For multi-modal pre-training of ECGs with reports using the **MIMIC-IV-ECG** dataset:
```shell script
$ python fairseq_signals/data/ecg_text/preprocess/preprocess_mimic_iv_ecg.py \
   /path/to/mimic-iv-ecg \
   --dest /path/to/output \
```

For ECG Question Answering task with the ECG-QA dataset:
* Map `ecg_id` to the corresponding ECG file path (you can find these scripts in the [ECG-QA repository](https://github.com/Jwoo5/ecg-qa))
    * For PTB-XL-based ECG-QA:
        ```shell script
        $ python mapping_ptbxl_samples.py ecgqa/ptbxl \
            --ptbxl-data-dir $ptbxl_dir \
            --dest $dest_dir
        ```
    * For MIMIC-IV-ECG-based ECG-QA:
        ```shell script
        $ python mapping_mimic_iv_ecg_samples.py ecgqa/mimic-iv-ecg \
            --mimic-iv-ecg-data-dir $mimic_iv_ecg_dir \
            --dest $dest_dir
        ```
* Preprocess ECG-QA and prepare manifests
    ```shell script
    $ python fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa.py /path/to/ecgqa \
        --dest /path/to/output \
        --apply_paraphrase
    ```

You don't need to run additional scripts to prepare manifest files for ECG-QA dataset since it automatically generates manifest files during the pre-processing process.

### Prepare data manifest
Given a directory that contains pre-processed PTB-XL data:
```shell script
$ python fairseq_signals/data/ecg_text/preprocess/manifest.py \
    /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $valid
```
Please find more details about pre-processing and data manifest [here](fairseq_signals/data/ecg_text/preprocess/README.md).

## Evaluation
To evaluate a trained model, run `fairseq-hydra-validate` with a configuration for evaluation (e.g., [`examples/scratch/ecg_classification/infer.yaml`](examples/scratch/ecg_classification/infer.yaml) or [`examples/scratch/ecg_question_answering/infer.yaml`](examples/scratch/ecg_question_answering/infer.yaml)):
```shell script
$ fairseq-hydra-validate \
    common_eval.path=/path/to/checkpoint.pt \
    task.data=/path/to/manifest \
    dataset.valid_subset=test \
    --config-dir examples/scratch/ecg_classification \
    --config-name infer
```
The model and task are restored from the checkpoint, and `task.data` should be the directory containing `$subset.tsv` for each subset in `dataset.valid_subset` (comma-separated, e.g., `valid,test`). It reports the loss and the metrics of the criterion used for training on each subset.

To save tensors from the model, add the following arguments (before `--config-dir`):
```shell script
    'common_eval.extract=[output,encoder_out,saliency]' \
    common_eval.results_path=/path/to/results \
```
* `output`: model outputs (logits) and targets, saved as `outputs_$subset.npy` and `targets_$subset.npy`
* `encoder_out`: encoder outputs, saved as `encoder_out_$subset.npy`
* `saliency`: attention weights of each transformer layer, saved as `saliency_$subset.npy`

An error is raised if the model does not support the requested tensor (e.g., `saliency` for non-transformer models). Note that `encoder_out` and `saliency` can be extracted only when all the samples have the same length, since the shape of each saved tensor is fixed when it is created.

To get the model outputs for data without labels, run `fairseq-hydra-inference` instead. It does not compute any metrics, but only saves the model outputs (logits) of each subset as `outputs_$subset.npy`:
```shell script
$ fairseq-hydra-inference \
    common_eval.path=/path/to/checkpoint.pt \
    task.data=/path/to/manifest \
    dataset.valid_subset=test \
    common_eval.results_path=/path/to/results \
    --config-dir examples/scratch/ecg_classification \
    --config-name infer
```

The rows of the saved tensors follow the order of the samples in `$subset.tsv`, also when running on multiple GPUs (`distributed_training.distributed_world_size`). The saved tensors are raw memory-mapped arrays, whose shape and dtype are stored in `*_header.pkl`. They cannot be loaded with `np.load`, so load them as follows:
```python
from fairseq_signals.utils.store import MemmapReader

outputs = MemmapReader.from_header("/path/to/results/outputs_test.npy").array
```
For a model with multiple outputs (e.g., `unet_cgm_segmentor`, which outputs both segmentation and classification results), each output is saved with its index (e.g., `outputs_$subset_0.npy`, `outputs_$subset_1.npy`).

### ECG segmentation
The loss and metrics of segmentation models are computed only for the annotated range of each sample (where `segment_mask` is `False`), so `common_eval.extract=[output]` is not supported for them. To get the outputs for all the time steps of each sample, run `fairseq-hydra-inference` with the configuration used for training (`common_eval.*` needs `+` since it is not written in the training configuration):
```shell script
$ fairseq-hydra-inference \
    +common_eval.path=/path/to/checkpoint.pt \
    task.data=/path/to/manifest \
    dataset.valid_subset=test \
    +common_eval.results_path=/path/to/results \
    --config-dir examples/scratch/ecg_segmentation/unet \
    --config-name unet3plus_aug
```
It saves `outputs_test.npy` with the shape of `(N, length, 4)`, where the last dimension corresponds to P wave, QRS complex, T wave, and none. For the configurations with CGM (e.g., `unet3plus_cgm`), the segmentation outputs are saved as `outputs_test_0.npy`, and the classification outputs as `outputs_test_1.npy`.

## Examples
We provide detailed READMEs for each model implementation:
* [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](examples/m3ae/README.md)
* [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](examples/medvill/README.md)
* [Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram](examples/w2v_cmsc/README.md)*
* [3KG: Contrastive Learning of 12-Lead Electrocardiograms using Physiologically-Inspired Augmentations](examples/3kg/README.md)
* [CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients](examples/cmsc/README.md)
* [wav2vec 2.0: A Framework for Self-supervised Learning of Speech Representations](examples/wav2vec2/README.md)
* [A Simple Framework for Contrastive Learning of Visual Representations](examples/simclr/README.md)

\* denotes for an official implementation

To train models from scratch (i.e., without pre-training) for each downstream task (ECG classification, patient identification, ECG question answering, and ECG segmentation), see [here](examples/scratch/README.md).

To train a model with multiple GPUs, set `distributed_training.distributed_world_size` to the number of GPUs. Note that the fine-tuning configurations with `model.feature_grad_mult: 0.0` freeze the feature extractor of the pre-trained model, so add `distributed_training.find_unused_parameters=true` to fine-tune them with multiple GPUs.

# Contact
If you have any questions or recommendations, please contact us via an issue or an e-mail.
* ojw0123@kaist.ac.kr
