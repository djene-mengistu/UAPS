# Uncertainty-aware and Dynamically-mixed Pseudo-labels for Semi-supervised Defect Segmentation (UAPS)
In this study, we propose a novel uncertainty-aware pseudo labels, which are generated from dynamically mixed predictions of multiple decoders that leverage a shared encoder network. The estimated uncertainty guides the pseudo-label-based supervision and regularizes the training when using the unlabeled samples. In our experiments on four public datasets for defect segmentation, the proposed method outperformed the fully supervised baseline and six state-of-the-art semi-supervised segmentation methods. We also conducted an extensive ablation study to demonstrate the effectiveness of our approach in various settings.
## Python >= 3.6
PyTorch >= 1.1.0
PyYAML, tqdm, tensorboardX
## Data Preparation
Download datasets. There are 3 datasets to download:
* NEU-SEG dataset from [NEU-seg](https://ieeexplore.ieee.org/document/8930292)
* DAGM dataset from [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* MT (Magnetic Tiles) dataset from [MTiles](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)

Put downloaded data into the following directory structure:
* data/
    * NEU_data/ ... # raw data of NEU-Seg
    * DAGM_data/ ...# raw data of DAGM
    * MTiles_data/ ...# raw data of MTiles
## Code usage
The training files and settings for each compared network is presented in separate directory. Train each network and test from the presented directory.
To train the proposed **UAPS** method run the following after setting hyperparameters such as labeled-ratio, iteration-per-epoch, consistency ramp length, and pair-wise-similarity loss coefficient.
```bash
python UAPS_train.py
```

To test the performance of the proposed method:
```bash
run UAPS_Testing.ipynb
```

To evaluate and visualize the pairwise similarirty map:
```bash
run UAPS_evalaute.ipynb
```
Similarly, train the proposed method, **simEps**, for the other datasets from the indicated directories after setting appropriate hyper-parametres.
## Some results and visualization
The results of the proposed method compared with the supervised baseline is presented as follows:

 
## Acknowledgment

This repo borrowed many implementations from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Contact
For any issue please contact me at djene.mengistu@gmail.com
