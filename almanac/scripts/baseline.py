import numpy as np
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from src.dataset.dataset import MultiInputDataset
from src.scoring.scoring_metrics import *
from src.utils.utils import save_evaluation_results


def get_baseline_results(output_filepath='../results/model_evaluation_results_cellminercdb.csv'):
    """Calculate performance scores for a baseline model that always predicts the mean of the training set"""
    train_dataset = MultiInputDataset(
        '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_train.csv.gz',
        'COMBOSCORE')
    mean = np.mean(train_dataset.y)
    del train_dataset
    test_dataset = MultiInputDataset(
        '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
        'COMBOSCORE')
    predictions = np.full(shape=test_dataset.y.shape, fill_value=mean)
    mse = MeanSquaredError()
    mse.update_state(test_dataset.y, predictions)
    rmse = RootMeanSquaredError()
    rmse.update_state(test_dataset.y, predictions)
    results_dict = {'keras_pearson': pearson(test_dataset.y, predictions), # going to be NaN because y_pred is constant
                    'keras_r2_score': r2_score(test_dataset.y, predictions), # it's practically 0, as expected
                    'keras_spearman': spearman(test_dataset.y, predictions), # going to be NaN because y_pred is constant
                    'mean_squared_error': mse.result().numpy(),
                    'root_mean_squared_error': rmse.result().numpy()}
    save_evaluation_results(results_dict=results_dict,
                            hyperparams={},
                            model_descr='Baseline (always predicts the mean of the training set)',
                            model_dir=np.nan,
                            output_filepath=output_filepath)

if __name__ == '__main__':
    get_baseline_results()