# Drug synergy prediction tool

This repository contains code for the paper _"A systematic evaluation of deep learning methods for the
prediction of drug synergy in cancer"_, by Delora Baptista, Pedro G. Ferreira and Miguel Rocha.

The software package allows users to preprocess multiomics datasets and build multimodal drug response prediction models using Tensorflow/Keras.
It provides access to reusable model architectures, as well as separate feature-encoding subnetworks that can be combined
in different ways to create new models. It also provides classes to load multi-input data for deep learning models, to
preprocess drug response, compound, and omics data, to perform hyperparameter optimization for both
deep learning and machine learning models, and to interpret deep learning models using deep learning-specific feature attribution algorithms from SHAP. 

![](https://github.com/BioSystemsUM/drug_response_pipeline/blob/master/Fig1.png)

## Requirements
- python
- pandas=1.2.3
- tensorflow=2.2.0
- deepchem=2.5.0
- spektral=1.0.7
- Ray Tune=1.0.1
- hpbandster 
- ConfigSpace
- tune-sklearn
- scikit-optimize
- rdkit=2020.09.1.0
- pubchempy
- chembl_structure_pipeline
- umap-learn=0.5.1
- scikit-learn=0.22.1
- xgboost=1.4.0
- lightgbm=3.2.1
- shap=0.39.0
- matplotlib
- seaborn
- dill
- pyyaml


## Installation
The code can be installed as a Python package:
```bash
pip install git+https://github.com/BioSystemsUM/drug_response_pipeline.git
``` 

### Docker
A Dockerfile replicating the environment in which our analyses were performed is also provided:
1. Install [docker](https://docs.docker.com/install/).
2. Clone the repository and build a Docker image:
```bash
git clone https://github.com/BioSystemsUM/drug_response_pipeline.git

docker build --squash -t drpred_tool -f docker/Dockerfile .
``` 

## Predicting drug synergy in cancer cell lines
Scripts to reproduce our analyses for the NCI ALMANAC
dataset are provided in the 'almanac' folder.

## Citation
```
@article{10.1371/journal.pcbi.1010200,
    doi = {10.1371/journal.pcbi.1010200},
    author = {Baptista, Delora AND Ferreira, Pedro G. AND Rocha, Miguel},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {A systematic evaluation of deep learning methods for the prediction of drug synergy in cancer},
    year = {2023},
    month = {03},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pcbi.1010200},
    pages = {1-26},
    abstract = {One of the main obstacles to the successful treatment of cancer is the phenomenon of drug resistance. A common strategy to overcome resistance is the use of combination therapies. However, the space of possibilities is huge and efficient search strategies are required. Machine Learning (ML) can be a useful tool for the discovery of novel, clinically relevant anti-cancer drug combinations. In particular, deep learning (DL) has become a popular choice for modeling drug combination effects. Here, we set out to examine the impact of different methodological choices on the performance of multimodal DL-based drug synergy prediction methods, including the use of different input data types, preprocessing steps and model architectures. Focusing on the NCI ALMANAC dataset, we found that feature selection based on prior biological knowledge has a positive impact—limiting gene expression data to cancer or drug response-specific genes improved performance. Drug features appeared to be more predictive of drug response, with a 41% increase in coefficient of determination (R2) and 26% increase in Spearman correlation relative to a baseline model that used only cell line and drug identifiers. Molecular fingerprint-based drug representations performed slightly better than learned representations—ECFP4 fingerprints increased R2 by 5.3% and Spearman correlation by 2.8% w.r.t the best learned representations. In general, fully connected feature-encoding subnetworks outperformed other architectures. DL outperformed other ML methods by more than 35% (R2) and 14% (Spearman). Additionally, an ensemble combining the top DL and ML models improved performance by about 6.5% (R2) and 4% (Spearman). Using a state-of-the-art interpretability method, we showed that DL models can learn to associate drug and cell line features with drug response in a biologically meaningful way. The strategies explored in this study will help to improve the development of computational methods for the rational design of effective drug combinations for cancer therapy.},
    number = {3},

}
```
