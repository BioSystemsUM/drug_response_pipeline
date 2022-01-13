# Drug response prediction tool

This repository contains code to preprocess multiomics datasets and to build multimodal drug response prediction models using Tensorflow/Keras.
It provides access to reusable model architectures, as well as separate feature-encoding subnetworks that can be combined
in different ways to create new models. It also provides classes to load multi-input data for deep learning models, to
preprocess drug response, compound, and omics data, to perform hyperparameter optimization for both
deep learning and machine learning models, and to interpret deep learning models using deep learning-specific feature attribution algorithms from SHAP. 

## Installation
The code can be installed as a Python package:
```bash
pip install git+https://gitlab.bio.di.uminho.pt/dlrsb/drug_response_pipeline.git
``` 
### Docker
A Dockerfile replicating the environment in which our analyses were performed is also provided:
1. Install [docker](https://docs.docker.com/install/).
2. Clone the repository and build a Docker image:
```bash
git clone https://gitlab.bio.di.uminho.pt/dlrsb/drug_response_pipeline.git

docker build --squash -t drpred_tool -f docker/Dockerfile .
``` 

## "Predicting drug synergy in cancer cell lines"
Scripts to reproduce our analyses for the ALMANAC
dataset are provided in the 'almanac' folder.