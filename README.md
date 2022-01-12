This repository has been extended from [Fairseq](https://github.com/pytorch/fairseq) to process ECG data, instead of audio data.


We provide implementatinos of methods for time-series data, especially for ECG:
* [CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients](https://arxiv.org/pdf/2005.13249.pdf)
* [3KG: Contrastive Learning of 12-Lead Electrocardiograms using Physiologically-Inspired Augmentations](https://arxiv.org/pdf/2106.04452.pdf)

# Requirments and Installation
* [PyTorch](https://pytorch.org) version >= 1.8.0
* Python version >= 3.6
* **To build cython components**: `python setup.py build_ext --inplace`

# Datasets
The datasets can be downloaded at this links:
* [PhysioNet 2021](https://moody-challenge.physionet.org/2021/): 
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/)

## Preprocess
...