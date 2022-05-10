import os
import argparse
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv, GlobalSumPool, GATConv
from deepchem.models.layers import Highway, DTNNEmbedding

from src.dataset.dataset import MultiInputDataset
from src.scoring import scoring_metrics
from src.utils.utils import save_evaluation_results


def predict_evaluate(settings_file, saved_model_path, output_path):
    """
    Predict test set using a saved model and calculate scores.

    Parameters
    ----------
    settings_file: str
        Path to the settings file that was originally used.
    saved_model_path: str
        Path to the saved model.
    output_path: str
        Path to file where the results will be written.
    """

    # Load settings file
    with open(settings_file, 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if 'gpu_to_use' in settings:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['gpu_to_use'])[1:-1]
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)

    # Load test data
    test_dataset = MultiInputDataset(settings['test_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                     output_col='COMBOSCORE')
    test_dataset.load_drugA(settings['test_drugA_filepath'])
    test_dataset.load_drugB(settings['test_drugB_filepath'])
    test_dataset.load_expr(settings['test_expr_filepath'])
    test_dataset.load_mut(settings['test_mut_filepath'])
    test_dataset.load_cnv(settings['test_cnv_filepath'])
    test_dataset.load_graph_data(nodes_file=settings['test_drugA_atom_feat_filepath'],
                                 adj_file=settings['test_drugA_adj_filepath'])
    test_dataset.load_graph_data(nodes_file=settings['test_drugB_atom_feat_filepath'],
                                 adj_file=settings['test_drugB_adj_filepath'])

    # Load saved model
    custom_objects={'keras_r2_score': scoring_metrics.keras_r2_score,
                    'keras_spearman': scoring_metrics.keras_spearman,
                    'keras_pearson': scoring_metrics.keras_pearson}
    if saved_model_path == '../results/2021-07-14_22-20-49/train_set_model':
        custom_objects['GCNConv'] = GCNConv
        custom_objects['GlobalSumPool'] = GlobalSumPool
    elif saved_model_path == '../results/2021-07-14_22-20-12/train_set_model.h5':
        custom_objects['DTNNEmbedding'] = DTNNEmbedding
        custom_objects['Highway'] = Highway
        custom_objects['math'] = tf.math
    elif saved_model_path == '../results/2021-09-10_13-40-49/train_set_model':
        custom_objects['GATConv'] = GATConv
        custom_objects['GlobalSumPool'] = GlobalSumPool

    model = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects)

    # Predict
    y_pred = np.squeeze(model.predict(test_dataset.X_dict, batch_size=64))

    # Save predictions
    df = pd.DataFrame(data={'experiment': test_dataset.get_row_ids(sep='+'),
                            'y_true': test_dataset.y,
                            'y_pred': y_pred})
    model_dir = os.path.split(saved_model_path)[0]
    df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)

    # Evaluate
    scores_dict = {}
    metrics = ['mean_squared_error', 'pearson', 'r2_score', 'root_mean_squared_error', 'spearman']
    for metric_name in metrics:
        if metric_name == 'root_mean_squared_error':
            scores_dict[metric_name] = scoring_metrics.mean_squared_error(test_dataset.y, y_pred, squared=False)
        else:
            metric_func = getattr(scoring_metrics, metric_name)
            scores_dict[metric_name] = metric_func(test_dataset.y, y_pred)

    save_evaluation_results(results_dict=scores_dict, hyperparams={}, model_descr=settings['model_description'],
                            model_dir=model_dir, output_filepath=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using saved models')
    parser.add_argument('-s',
                        '--settings-file',
                        type=str,
                        help='Path to the settings file (YAML format) that was used to setup the model.')
    parser.add_argument('-m',
                        '--saved-model-path',
                        type=str,
                        help='Path to the saved model.')
    parser.add_argument('-o',
                        '--output-path',
                        type=str,
                        help='Output filepath.')
    args = vars(parser.parse_args())
    print(args)
    predict_evaluate(**args)