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
* Python version >= 3.6, and <= 3.9
* PIP version <= 24.0; if your pip version is higher than 24.0, please run:
    ```bash
    pip install pip==24.0
    ```
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
git clone https://github.com/Jwoo5/fairseq-signals
cd fairseq-signals
pip install --editable ./
```

* **To preprocess ECG datasets**: `pip install pandas scipy wfdb`
* **To build cython components**: `python setup.py build_ext --inplace`
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`

# Getting Started
## For uni-modal tasks (ECG Classification, ...)
### Prepare ECG dataset
We provide pre-processing codes for various ECG datasets.

* [PhysioNet2021](https://physionet.org/content/challenge-2021/1.0.3/#files)
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)

### Pre-process
Given a directory that contains WFDB directories to be pre-processed for **PhysioNet2021**:

```shell script
$ python fairseq_signals/data/ecg/preprocess/preprocess_physionet2021.py \
    /path/to/physionet2021/ \
    --dest /path/to/output \
    --workers $N
```

Given a directory that contains .dat files from PTB-XL:
```shell script
$ python fairseq_signals/data/ecg/preprocess/preprocess_ptbxl.py \
    /path/to/ptbxl/records500/ \
    --dest /path/to/output
```

### Prepare data manifest
Given a directory that contains pre-processed data:
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest.py \
    /path/to/data/ \
    --dest /path/to/manifest \
    --valid-percent $valid
```
For patient identification:
```shell script
$ python fairseq_signals/data/ecg/preprocess/manifest_identification.py \
    /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $valid
```
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
    $ fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa.py /path/to/ecgqa \
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

# Contact
If you have any questions or recommendations, please contact us via an issue or an e-mail.
* ojw0123@kaist.ac.kr
