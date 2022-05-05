import numpy as np
import pandas as pd

from src.scoring import scoring_metrics
from src.utils.utils import save_evaluation_results


def calculate_ensemble_scores(paths_to_saved_predictions, ensemble_description):
    y_true = None
    base_model_predictions = []
    for pred_file in paths_to_saved_predictions:
        df = pd.read_csv(pred_file)
        if y_true is None:
            y_true = df['y_true'].values
            experiments = df['experiment']
        base_model_predictions.append(df['y_pred'].values)

    ensemble_pred = np.mean(np.array(base_model_predictions), axis=0)

    ensemble_pred_df = pd.DataFrame(data={'experiment': experiments,
                            'y_true': y_true,
                            'y_pred': ensemble_pred})
    ensemble_pred_df.to_csv('../results/ensemble_predictions.csv')

    scores_dict = {}
    metrics = ['mean_squared_error', 'pearson', 'r2_score', 'root_mean_squared_error', 'spearman']
    for metric_name in metrics:
        if metric_name == 'root_mean_squared_error':
            scores_dict[metric_name] = scoring_metrics.mean_squared_error(y_true, ensemble_pred, squared=False)
        else:
            metric_func = getattr(scoring_metrics, metric_name)
            scores_dict[metric_name] = metric_func(y_true, ensemble_pred)
    print(scores_dict)

    save_evaluation_results(scores_dict, hyperparams={},
                            model_descr=ensemble_description,
                            model_dir='',
                            output_filepath='../results/ensemble_evaluation_results.csv')

if __name__ == '__main__':
    saved_predictions = ['../results/2021-07-06_12-19-03/predictions.csv',  # expr (DGI) + drugs (ECFP4)
                         '../results/2021-07-06_12-02-29/predictions.csv',  # expr (landmark) + drugs (ECFP4)
                         '../results/2021-07-06_12-12-35/predictions.csv',  # expr (COSMIC) + drugs (ECFP4)
                         '../results/2021-09-09_15-05-24/predictions.csv', # expr (DGI) + mut (pathway-level) + cnv (DGI) + drugs (ECFP4)
                         '../results/2021-07-13_20-03-54/predictions.csv',  # expr (DGI) + drugs (LayeredFP)
                         '../results/2021-07-10_14-02-58/predictions.csv', # expr (protein coding, clustering order 1D CNN) + drugs (ECFP4)
                         '../results/2021-07-13_15-43-50/predictions.csv',  # expr (DGI) + drugs (MTE)
                         '../results/2021-07-14_22-20-49/predictions.csv',  # expr (DGI) + drugs (GCN)
                         '../results/2021-07-11_13-02-09/predictions.csv', # expr (protein coding) + drugs (ECFP4)
                         '../results/2021-07-06_12-21-29/predictions.csv',  # expr (NCG) + drugs (ECFP4)
                         '../results/2021-10-26_14-36-02/predictions.csv',  # LGBM
                         '../results/2021-09-27_12-58-08/predictions.csv',  # XGBoost
                         '../results/2021-09-27_23-01-20/predictions.csv'  # RF
                         ]
    calculate_ensemble_scores(paths_to_saved_predictions=saved_predictions,
                              ensemble_description='Voting Ensemble - Top 10 DL models +LGBM+XGBoost+RF')
