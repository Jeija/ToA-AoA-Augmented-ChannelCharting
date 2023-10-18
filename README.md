# Augmenting Channel Charting with Classical Wireless Source Localization Techniques

This repository contains the source code for the paper

> Florian Euchner, Phillip Stephan, Stephan ten Brink: "Augmenting Channel Charting with Classical Wireless Source Localization Techniques"

presented at the Asilomar Conference on Signals, Systems, and Computers in November 2023.

### Prerequesites
Our code is based on Python, TensorFlow, NumPy, SciPy and Matplotlib.
Source files are provided as Jupyter Notebooks, which can be opened directly here on GitHub or using e.g. [https://jupyter.org/](JupyterLab).

We run our Channel Charting experiments on a JupyterHub server with NVMe storage, AMD EPYC 7262 8-Core Processor, 64GB RAM, and a NVIDIA GeForce RTX 4080 GPU for accelerating TensorFlow.
All indications of computation times are measured on this system.
It should also be possible to run our notebooks on less performant systems.

### Download Datasets
As a very first step, use `0_DownloadDatasets.ipynb` to download parts of the [dichasus-cf0x](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/) dataset that we use for training and testing.
Due to the large file size, this dataset is not included in this git repository.

### File Description

### Reference
```
@inproceedings{euchner2023augmenting,
	author    = {Euchner, Florian and Stephan, Phillip and ten Brink, Stephan},
	title     = {{Augmenting Channel Charting with Classical Wireless Source Localization Techniques}},
	booktitle = {Asilomar Conference on Signals, Systems, and Computers},
	year      = {2023}
}
```