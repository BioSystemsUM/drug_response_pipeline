import pickle
import numpy as np
from tensorflow import keras

import scoring_metrics

class VotingEnsemble(object):
	"""
	An ensemble meta-estimator that combines the predictions of
	various models. For classification tasks, this is a majority
	voting ensemble. For regression tasks, it averages the
	individual predictions to obtain the final prediction.
	"""

	# it could inherit from Model, but that might be confusing as self.model would be a list of Model objects instead of a single model

	def __init__(self, base_models, mode):
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
		Fits the models using the datasets provided.
		"""
		# dataset format will be different depending on the type of model (Keras multi-input vs ML models)
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
		Makes predictions for the provided dataset.
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
		Evaluates the performance of this ensemble on the specified datasets.
		"""
		predictions = self.predict(datasets)
		scores_dict = {}
		for metric_name in metrics:
			metric_func = getattr(scoring_metrics, metric_name)
			scores_dict[metric_name] = metric_func(datasets[0].y, predictions)
		return scores_dict