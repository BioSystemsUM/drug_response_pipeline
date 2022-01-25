# Predicting drug synergy in cancer cell lines

## Data
ALMANAC drug response data in the form of ComboScores for <cell line, drugA, drugB> triplets were
and RNA-Seq data for the NCI-60 cell lines were downloaded from [CellMinerCDB](https://discover.nci.nih.gov/rsconnect/cellminercdb/).

Mutation and copy number variation data were downloaded from [CBioportal](https://www.cbioportal.org/study/summary?id=cellline_nci60).

An SDF file mapping compound identifiers to SMILES strings ([ComboCompoundSet.sdf](https://wiki.nci.nih.gov/download/attachments/338237347/ComboCompoundSet.sdf?version=1&modificationDate=1493822360000&api=v2)) and a file mapping NSC identifiers to compound names ([ComboCompoundNames_small.txt](https://wiki.nci.nih.gov/download/attachments/338237347/ComboCompoundNames_small.txt?version=1&modificationDate=1493822467000&api=v2)) were obtained from DTP.

The preprocessed response dataset and filtered gene expression, mutation and CNV files (before merging with the response 
dataset) can be downloaded from https://nextcloud.bio.di.uminho.pt/s/p5RXgkQwH5HoEiw. 
This repository also contains the fully preprocessed drug and gene expression data required to run the 
expr<sub>DGI</sub> + drugs<sub>ECFP4</sub> model described in our study.  

## Scripts
The original drug response and omics files from CellMinerCDB and CBioportal can be preprocessed by running the *preprocess_almanac.py* script.

The fully preprocessed and split drugA and drugB files can be created by running the *create_drug_files.py* script and 
specifying the desired featurization method, as in the following example:
```bash
python create_drug_files.py --featurization-type ECFP4
``` 

The fully preprocessed and split omics files can be created by running the *create_omics_files.py* script, specifying 
the name of the corresponding feature-encoding subnetwork that will be used in the model, as in the following example:
```bash
python create_omics_files.py --expr-subnetwork-type 'expr (DGI)'
``` 

Deep learning models can be trained and evaluated using the *train_keras.py* script and a configuration file: 
```bash
python train_keras.py settings_files/expr_targets_full_minmaxscaler_drug_dense.yml
``` 

Machine learning models can be trained and evaluated in a similar manner:
```bash
python train_sklearn.py settings_files/ml/expr_targets_drug_ecfp4_lgbm.yml
``` 

Our SHAP analysis was performed using the *interpret_almanac_model.py* script.


## Results
The 'results' folder contains the full results tables (with results for several scoring metrics and information on the 
tuned hyperparameters), the plots that were generated from these tables, and the results from the SHAP analysis, 
including the saved Explanation object and a CSV file containing the calculated SHAP values.

Note: In our results files, the drug-gene interactions (DGI) that were used to filter the omics files were originally 
called "targets". 

## Trained models
Our trained models can be downloaded from https://nextcloud.bio.di.uminho.pt/s/Tj2NT2kMGKXtZaq
