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
@article {Baptista2022.05.16.492054,
	author = {Baptista, Delora and Ferreira, Pedro G. and Rocha, Miguel},
	title = {A systematic evaluation of deep learning methods for the prediction of drug synergy in cancer},
	elocation-id = {2022.05.16.492054},
	year = {2022},
	doi = {10.1101/2022.05.16.492054},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {One of the main obstacles to the successful treatment of cancer is the phenomenon of drug resistance. A common strategy to overcome resistance is the use of combination therapies. However, the space of possibilities is huge and efficient search strategies are required. Machine Learning (ML) can be a useful tool for the discovery of novel, clinically relevant anti-cancer drug combinations. In particular, deep learning (DL) has become a popular choice for modeling drug combination effects. Here, we set out to examine the impact of different methodological choices on the performance of multimodal DL-based drug synergy prediction methods, including the use of different input data types, preprocessing steps and model architectures. Focusing on the NCI ALMANAC dataset, we found that feature selection based on prior biological knowledge has a positive impact on performance. Drug features appeared to be more predictive of drug response. Molecular fingerprint-based drug representations performed slightly better than learned representations, and gene expression data of cancer or drug response-specific genes also improved performance. In general, fully connected feature-encoding subnetworks outperformed other architectures, with DL outperforming other ML methods. Using a state-of-the-art interpretability method, we showed that DL models can learn to associate drug and cell line features with drug response in a biologically meaningful way. The strategies explored in this study will help to improve the development of computational methods for the rational design of effective drug combinations for cancer therapy.Author summary Cancer therapies often fail because tumor cells become resistant to treatment. One way to overcome resistance is by treating patients with a combination of two or more drugs. Some combinations may be more effective than when considering individual drug effects, a phenomenon called drug synergy. Computational drug synergy prediction methods can help to identify new, clinically relevant drug combinations. In this study, we developed several deep learning models for drug synergy prediction. We examined the effect of using different types of deep learning architectures, and different ways of representing drugs and cancer cell lines. We explored the use of biological prior knowledge to select relevant cell line features, and also tested data-driven feature reduction methods. We tested both precomputed drug features and deep learning methods that can directly learn features from raw representations of molecules. We also evaluated whether including genomic features, in addition to gene expression data, improves the predictive performance of the models. Through these experiments, we were able to identify strategies that will help guide the development of new deep learning models for drug synergy prediction in the future.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/05/16/2022.05.16.492054},
	eprint = {https://www.biorxiv.org/content/early/2022/05/16/2022.05.16.492054.full.pdf},
	journal = {bioRxiv}
}
```
