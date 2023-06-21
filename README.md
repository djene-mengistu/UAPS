# Uncertainty-aware and Dynamically-mixed Pseudo-labels for Semi-supervised Defect Segmentation (UAPS)
In this study, we propose a novel uncertainty-aware pseudo labels, which are generated from dynamically mixed predictions of multiple decoders that leverage a shared encoder network. The estimated uncertainty guides the pseudo-label-based supervision and regularizes the training when using the unlabeled samples. In our experiments on four public datasets for defect segmentation, the proposed method outperformed the fully supervised baseline and six state-of-the-art semi-supervised segmentation methods. We also conducted an extensive ablation study to demonstrate the effectiveness of our approach in various settings.

The illustration of the uncertainty map and error is presented as follows:
<p align="center">
<img src="/fig_data/uncertainty.jpg" width="40%" height="10%">
</p>

**Fig. 1:** Illustration of the network prediction, prediction error which is false negatives and false positives, and the uncertainty map computed as the KL-distance between primary
decoder and average predictions.

The overall architecture of the proposed method is presented as follows:
<p align="center">
<img src="/fig_data/main-arch.jpg" width="60%" height="40%">
</p>

**Fig. 2:** Illustration of the overall framework of the UAPS using the unlabeled samples. We apply various perturbations to the encoder output to obtain different predictions with multiple decoders. The predictions from the decoders are then combined using randomly generated weights to compute a pseudo-label. We calculate uncertainty as the KL-distance between the average prediction and each decoder’s prediction.

The baseline architecture and the perterbation method is presented as follows:
<p align="center">
<img src="/fig_data/UNET-baseline.jpg" width="60%" height="40%">
</p>

**Fig. 3:** Illustration of baseline architecture adapted from U-Net with perturbation module. The encoder output remains unchanged at each block, while the type of perturbation changes to produce different versions of the outputs to be fed to the decoder networks.

# Full paper source:
You can read the details about the methods, implementation, and results from: (https://ieeexplore.ieee.org/document/9994033)

**Please cite ourwork as follows:**
```
@article{sime2022semi,
  title={Semi-Supervised Defect Segmentation with Pairwise Similarity Map Consistency and Ensemble-Based Cross-Pseudo Labels},
  author={Sime, Dejene M and Wang, Guotai and Zeng, Zhi and Wang, Wei and Peng, Bei},
  journal={IEEE Transactions on Industrial Informatics},
  year={2022},
  publisher={IEEE}
}
```

## Python >= 3.6
PyTorch >= 1.1.0
PyYAML, tqdm, tensorboardX
## Data Preparation
Download datasets. There are 4 datasets to download:
* NEU-SEG dataset from [NEU-seg](https://ieeexplore.ieee.org/document/8930292)
* DAGM dataset from [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* MT (Magnetic Tiles) dataset from [MTiles](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)
* KoSDD2 (KolektorSDD2) dataset from [KoSDD2](https://www.vicos.si/resources/kolektorsdd2/)

Put downloaded data into the following directory structure:
* data/
    * NEU_data/ ... # raw data of NEU-Seg
    * DAGM_data/ ...# raw data of DAGM
    * MTiles_data/ ...# raw data of MTiles
    * KoSDD2_data/ ...# raw data of KoSDD2
## Code usage
The training files and settings for each compared network is presented in separate directory. Train each network and test from the presented directory.
To train the proposed **UAPS** method run the following after setting hyperparameters such as labeled-ratio, iteration-per-epoch, consistency ramp length, and consistency loss coefficients.
```bash
python UAPS_train.py
```

To test the performance of the proposed method:
```bash
run UAPS_Testing.ipynb
```

Similarly, train the proposed method, **UAPS**, for the other datasets from the indicated directories after setting appropriate hyper-parametres.
## Some results and visualizations
Visualization of the segmetnation results on selected samples are presented as follows:
<p align="center">
<img src="/fig_data/neu-viz.jpg" width="60%" height="50%">
</p>

**Fig. 4:** Visualization of the segmentation results on the NEU-Seg dataset. The regions indicated by the dashed-red-box shows wrong prediction.

<p align="center">
<img src="/fig_data/mt-viz.jpg" width="60%" height="50%">
</p>

**Fig. 5:** Visualization of the segmentation results on the MTiles dataset.

## Ablation experiments
Results from different model settings are presented as follows:
<p align="center">
<img src="/fig_data/loss-effects.jpg" width="40%" height="20%">
</p>

**Fig. 6:** Effects of different loss combinations.

<p align="center">
<img src="/fig_data/effects_of_loss_coef.jpg" width="40%" height="20%">
</p>

**Fig. 7:** Effects of loss coefficents.

<p align="center">
<img src="/fig_data/dynamic-mixing.jpg" width="40%" height="20%">
</p>

**Fig. 8:** Proposed dynamic-mixing vs. Averaging to generate pseudo-labels.

<p align="center">
<img src="/fig_data/decoder-effect.jpg" width="40%" height="20%">
</p>

**Fig. 9:** Effects of number of auxuliary decoders on segmentation performance and inference time.

 
## Acknowledgment

This repo borrowed many implementations from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [simEps](https://github.com/djene-mengistu/simEps/tree/main)

## Contact
For any issue please contact me at djene.mengistu@gmail.com
