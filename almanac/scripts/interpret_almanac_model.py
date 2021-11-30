import os

from tensorflow import keras
#tf.compat.v1.disable_v2_behavior() # https://github.com/slundberg/shap/issues/1055#issuecomment-708504795 and https://github.com/slundberg/shap/issues/1286
import numpy as np
import pandas as pd

from src.dataset.dataset import MultiInputDataset
from src.interpretability.interpret import ModelInterpreter

from src.scoring.scoring_metrics import keras_r2_score, keras_spearman, keras_pearson


os.environ['CUDA_VISIBLE_DEVICES'] = "3"
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def shap_analysis():
    print('loading training data')
    train_dataset = MultiInputDataset(response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_train.csv.gz',
                                      id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                      output_col='COMBOSCORE',
                                      input_order=['expr', 'drugA', 'drugB'])
    train_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_train.npy')
    train_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_train.npy')
    train_dataset.load_expr('../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_train.npy')

    print('loading test data')
    test_dataset = MultiInputDataset(response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
                                     id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                     output_col='COMBOSCORE',
                                     input_order=['expr', 'drugA', 'drugB'])
    test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
    test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
    test_dataset.load_expr('../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')
    test_dataset.load_feature_names(feature_names_filepath='../data/nci_almanac_preprocessed/targets_ecfp4_model_feature_names.pkl')

    # Save predictions to file for later
    model = keras.models.load_model('../results/2021-07-06_12-19-03/train_set_model.h5',
                                    custom_objects={'keras_r2_score': keras_r2_score,
                                                    'keras_spearman': keras_spearman,
                                                    'keras_pearson': keras_pearson})
    y_pred = np.squeeze(model.predict(test_dataset.X_dict))
    df = pd.DataFrame(data={'experiment': test_dataset.get_row_ids(sep='+'),
                            'y_true': test_dataset.y,
                            'y_pred': y_pred})
    df.to_csv('../results/2021-07-06_12-19-03/predictions.csv', index=False)

    # Explain model predictions
    print('computing shap values')
    interpreter = ModelInterpreter(explainer_type='Deep',
                                   saved_model_path='../results/2021-07-06_12-19-03/train_set_model.h5',
                                   dataset=test_dataset)
    interpreter.compute_shap_values(train_dataset=train_dataset,
                                    n_background_samples=1000)

    print('saving shap values and explanation objects')
    interpreter.save_shap_values(row_ids=test_dataset.get_row_ids(sep='+'),
                                 output_filepath='../results/shap_analysis/shap_values.csv')
    interpreter.save_explanation(output_filepath='../results/shap_analysis/shap_explanation.pkl', multi_input=False)
    interpreter.save_explanation(output_filepath='../results/shap_analysis/multi_input_shap_explanation.pkl', multi_input=True)

    # print('plotting shap values')
    interpreter = ModelInterpreter(explainer_type='Deep',
                                   saved_model_path=None,
                                   dataset=test_dataset)
    interpreter.load_explanation('../results/shap_analysis/shap_explanation.pkl', multi_input=False)
    interpreter.load_explanation('../results/shap_analysis/multi_input_shap_explanation.pkl', multi_input=True)
    data_types = ['all', 'expr', 'drugA', 'drugB']
    sample_ids = [11835, 10730, 11047, 11985, 14397, 20178, 11426, 11429, 1827, 25225]

    for data_type in data_types:
        interpreter.plot_feature_importance(plot_type='bar',
                                            input_type=data_type,
                                            max_display=20,
                                            output_filepath='../results/shap_analysis/feature_importance_barplot_%s.png' % data_type)

        interpreter.plot_feature_importance(plot_type='beeswarm',
                                            input_type=data_type,
                                            max_display=20,
                                            output_filepath='../results/shap_analysis/feature_importance_beeswarm_%s.png' % data_type)

        for sample_id in sample_ids:
            interpreter.plot_sample_explanation(row_index=sample_id, plot_type='waterfall', input_type=data_type, max_display=21,
                                            output_filepath='../results/shap_analysis/row%s_sample_explanation_waterfall_%s.png' % (sample_id, data_type))


if __name__ == '__main__':
    shap_analysis()

