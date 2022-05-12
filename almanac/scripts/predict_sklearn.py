import os
import pickle

import pandas as pd

from src.dataset.dataset import MultiInputDataset


def predict(model_path):
    print('loading test data')
    dgi_ecfp4_test_dataset = MultiInputDataset(
        response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
        id_cols=['CELLNAME', 'NSC1', 'NSC2'],
        output_col='COMBOSCORE')
    dgi_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
    dgi_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
    dgi_ecfp4_test_dataset.load_expr(
        '../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_dgi_minmaxscaled_test.npy')
    dgi_ecfp4_test_dataset.concat_features()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(dgi_ecfp4_test_dataset.X)

    df = pd.DataFrame(data={'experiment': dgi_ecfp4_test_dataset.get_row_ids(sep='+'),
                            'y_true': dgi_ecfp4_test_dataset.y,
                            'y_pred': y_pred})
    model_dir = os.path.split(model_path)[0]
    df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    paths_to_saved_models = ['../results/2021-10-26_14-36-02/train_model.pkl',  # LGBM
                             '../results/2021-09-27_12-58-08/train_model.pkl',  # XGBoost
                             '../results/2021-09-27_23-01-20/train_model.pkl',  # RF
                             ]
    for saved_model_path in paths_to_saved_models:
        predict(saved_model_path)