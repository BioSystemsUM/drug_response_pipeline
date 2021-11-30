import pickle
import copy

import numpy as np
import pandas as pd
from tensorflow import keras
import shap
import matplotlib.pyplot as plt
from src.interpretability.shap_plots_modified import bar, beeswarm, waterfall
from src.scoring.scoring_metrics import keras_r2_score, keras_spearman, keras_pearson


class ModelInterpreter(object):

    def __init__(self, explainer_type, saved_model_path, dataset):
        self.explainer_type = explainer_type
        if saved_model_path is not None:
            self.model = self._load_model(saved_model_path)
        else:
            self.model = None
        self.multi_input_shap_values = None
        self.shap_values = None
        self.dataset = dataset

    def _load_model(self, path_to_model):
        """Loads a Keras model from an HDF5 file or a SavedModel directory"""
        # if '.h5' in path_to_model:
        #     model = keras.models.load_model(path_to_model,
        #                                     custom_objects={'keras_r2_score': keras_r2_score,
        #                                                     'keras_spearman': keras_spearman,
        #                                                     'keras_pearson': keras_pearson})
        # else:
        #     model = keras.models.load_model(path_to_model)
        model = keras.models.load_model(path_to_model, custom_objects={'keras_r2_score': keras_r2_score,
                                                                       'keras_spearman': keras_spearman,
                                                                       'keras_pearson': keras_pearson})
        return model

    def compute_shap_values(self, train_dataset, n_background_samples=100):
        if self.explainer_type == 'Deep':
            shap.explainers._deep.deep_tf.op_handlers[
                "AddV2"] = shap.explainers._deep.deep_tf.passthrough  # so that it works for Keras models with Batch Normalization
            background = train_dataset.sample(n=n_background_samples).X_dict_to_list()
            del train_dataset
            explainer = shap.DeepExplainer(self.model, background)
            expected_value = explainer.expected_value
        elif self.explainer_type == 'Gradient':
            X_train_list = train_dataset.X_dict_to_list()
            explainer = shap.GradientExplainer(self.model, X_train_list)
            expected_value = self.model.predict(X_train_list).mean(0)

        shap_X = self.dataset.X_dict_to_list()
        shap_values = explainer.shap_values(shap_X)[0] # as we only have 1 output
        feature_names = self.dataset.feature_names
        base_values = np.tile(expected_value, self.dataset.get_n_rows())  # repeat value across all rows

        # Multi-input Shap values saved as a dict:
        self.multi_input_shap_values = {}
        for i, input_name in enumerate(self.dataset.input_order):
            self.multi_input_shap_values[input_name] = shap._explanation.Explanation(values=shap_values[i],
                                                                                     base_values=base_values,
                                                                                     data=self.dataset.X_dict[input_name],
                                                                                     feature_names=feature_names[input_name])
                                                                                     #instance_names=row_ids) # setting instance_names is not working

        # # All Shap values concatenated into a single array (for plotting, etc.):
        shap_values = np.concatenate(shap_values, axis=1)
        shap_X = self.dataset.concat_features()
        feature_names_concat = self.dataset.feature_names_to_list()
        self.shap_values = shap._explanation.Explanation(values=shap_values, base_values=base_values,
                                                         data=shap_X,
                                                         feature_names=feature_names_concat)
                                                         #instance_names=row_ids) # setting instance_names is not working
        return self.shap_values.values

    def plot_feature_importance(self, plot_type='beeswarm', input_type='all', max_display=20, output_filepath=None):
        if input_type == 'all':
            explanation_obj = copy.deepcopy(self.shap_values) # because beeswarm modifies the shap_values for some reason
        else:
            explanation_obj = copy.deepcopy(self.multi_input_shap_values[input_type])

        if plot_type == 'bar':
            #shap.plots.bar(explanation_obj, max_display=max_display, show=False)
            bar(explanation_obj, max_display=max_display, show=False)
        elif plot_type == 'beeswarm':
            #shap.plots.beeswarm(explanation_obj, max_display=max_display, show=False)
            beeswarm(explanation_obj, max_display=max_display, show=False)
        else:
            raise ValueError('Unsupported plot type')

        if output_filepath is not None:
            plt.tight_layout()
            plt.savefig(output_filepath)
        else:
            plt.show()
        plt.close()

    def plot_feature_effect(self, feature, output_filepath=None):
        shap.plots.scatter(self.shap_values[:, feature], show=False)
        #shap.plots.scatter(self.shap_values[:, feature], color=self.shap_values, show=False)
        if output_filepath is not None:
            plt.tight_layout()
            plt.savefig(output_filepath)
        else:
            plt.show()
        plt.close()

    def plot_sample_explanation(self, row_index, plot_type='waterfall', input_type='all', max_display=20,
                                output_filepath=None):
        if input_type == 'all':
            explanation_obj = copy.deepcopy(self.shap_values)
        else:
            explanation_obj = copy.deepcopy(self.multi_input_shap_values[input_type])

        if plot_type == 'bar':
            #shap.plots.bar(explanation_obj[row_index], max_display=max_display, show=False)
            bar(explanation_obj[row_index], max_display=max_display, show=False)
        elif plot_type == 'waterfall':
            #shap.plots.waterfall(explanation_obj[row_index], max_display=max_display, show=False)
            waterfall(explanation_obj[row_index], max_display=max_display, show=False)
        elif plot_type == 'force':
            shap.plots.force(explanation_obj[row_index], show=False, matplotlib=True)
        else:
            raise ValueError('Unsupported plot type.')

        if output_filepath is not None:
            plt.tight_layout()
            plt.savefig(output_filepath)
        else:
            plt.show()
        plt.close()

    def save_shap_values(self, row_ids, output_filepath):
        df = pd.DataFrame(data=self.shap_values.values, columns=self.shap_values.feature_names,
                          index=row_ids)
        df.to_csv(output_filepath)

    def save_explanation(self, output_filepath, multi_input=False):
        if multi_input:
            explanation = self.multi_input_shap_values
        else:
            explanation = self.shap_values
        with open(output_filepath, 'wb') as f:
            pickle.dump(explanation, f)

    def load_explanation(self, filepath, multi_input=False):
        with open(filepath, 'rb') as f:
            explanation = pickle.load(f)
        if multi_input:
            self.multi_input_shap_values = explanation
        else:
            self.shap_values = explanation
