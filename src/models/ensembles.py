import pickle
import numpy as np
from tensorflow import keras

from src.scoring import scoring_metrics

class VotingEnsemble(object):
	"""
	An ensemble meta-estimator that combines the predictions of
	various models. For classification tasks, this is a majority
	voting ensemble. For regression tasks, it averages the
	individual predictions to obtain the final prediction.
	"""

	def __init__(self, base_models, mode):
		"""
		Parameters
		----------
		base_models: list or None
			List of base models to include in the ensemble.
		mode: str
			The type of task ('regression' or 'classification').
		"""
		self.base_models = base_models
		self.mode = mode

	def load_models(self, model_paths, custom_objs):
		self.base_models = []
		for model_path, custom_objects in zip(model_paths, custom_objs):
			print(model_path)
			if '.pkl' in model_path:
				with open(model_path, 'rb') as f:
					model = pickle.load(f)
					self.base_models.append(model)
			else:
				model = keras.models.load_model(model_path, custom_objects=custom_objects)
				self.base_models.append(model)


	def fit(self, datasets, fit_params):
		"""
		Fit the base models using the dataset provided for each model.

		Parameters
		----------
		datasets: list
			A list of multi-input datasets to be used as input to the base models. Each model uses its own multi-input
			dataset because the types of input features may be different for each type of model (the output variable is
			the same for all). The length of the list must be the same as the length of base_models.
		fit_params: dict
			Parameters to pass to Keras Model's fit method

		Returns
		-------
		None

		"""
		fit_models = []
		for model, dataset in zip(self.base_models, datasets):
			if isinstance(model, keras.models.Model):
				model.fit(dataset.X_dict, y=dataset.y, batch_size=fit_params['batch_size'], epochs=fit_params['epochs']) # TODO: allow user to define more options to fit Keras models
			else:
				model.fit(dataset.X, y=dataset.y)
			fit_models.append(model)
		self.base_models = fit_models

	def predict(self, datasets):
		"""
		Predicts the output value for the provided data.

		Parameters
		----------
		datasets: list
			A list of multi-input datasets to be used as input to the base models. Each model uses its own multi-input
			dataset because the types of input features may be different for each type of model. The length of the list
			must be the same as the length of base_models.

		Returns
		-------
		ensemble_pred: array
			The predictions made by the ensemble.
		"""
		# dataset format will be different depending on the type of model (Keras multi-input vs ML models)
		predictions = []
		for model, dataset in zip(self.base_models, datasets):
			if isinstance(model, keras.models.Model):
				y_pred = np.squeeze(model.predict(dataset.X_dict))
			else:
				y_pred = model.predict(dataset.X)
			predictions.append(y_pred)

		if self.mode == 'classification':  # classification - majority class
			ensemble_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
			                                    axis=0, arr=np.array(predictions))
		else:  # regression - ensemble prediction is the average value of predictions
			ensemble_pred = np.mean(np.array(predictions), axis=0)
		return ensemble_pred

	def evaluate(self, datasets, metrics):
		"""
		Evaluates the performance of the ensemble.
		Parameters
		----------
		datasets: list
			A list of multi-input datasets to be used as input to the base models. Each model uses its own multi-input
			dataset because the types of input features may be different for each type of model (the output variable
			is the same for all). The length of the list must be the same as the length of base_models.
		metrics: list
			The metrics that will be used to score the ensemble.

		Returns
		-------
		scores_dict: dict
			A dictionary containing the performance scores.
		"""
		predictions = self.predict(datasets)
		scores_dict = {}
		for metric_name in metrics:
			metric_func = getattr(scoring_metrics, metric_name)
			scores_dict[metric_name] = metric_func(datasets[0].y, predictions)
		return scores_dict
