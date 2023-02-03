import pickle
import numpy as np


def input_type_importance():
    """Calculate mean absolute SHAP values for features grouped by input type."""
    # based on https://github.com/slundberg/shap/issues/1892
    # https://github.com/slundberg/shap/issues/933
    # https://github.com/slundberg/shap/issues/651
    # https://github.com/slundberg/shap/issues/282
    # https://github.com/slundberg/shap/issues/465
    # this seems to be what SHAP does when grouping features at the end of its barplots: https://github.com/slundberg/shap/blob/master/shap/plots/_bar.py

    with open('../results/shap_analysis/multi_input_shap_explanation.pkl', 'rb') as f:
        multi_input_shap = pickle.load(f)

    with open('../results/shap_analysis/input_type_importance.txt', 'w') as f:
        for key in multi_input_shap:
            sum = np.sum(multi_input_shap[key].values,
                         axis=1)  # group all columns by summing the columns
            sum_mean_abs = np.mean(np.abs(sum))  # get mean absolute SHAP value for the grouped features
            f.write('Mean absolute SHAP value for input type "%s": %s \n' % (key, str(sum_mean_abs)))
            print('Mean absolute SHAP value for input type "%s": %s' % (key, str(sum_mean_abs)))

if __name__ == '__main__':
    input_type_importance()