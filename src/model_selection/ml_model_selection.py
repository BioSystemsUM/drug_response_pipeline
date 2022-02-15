import os

import pandas as pd
from sklearn import clone
from sklearn.model_selection import PredefinedSplit
from tune_sklearn import TuneSearchCV


class MLRayTuneSearch(object):
    """Tune hyperparameters using Bayesian optimization with TuneSearchCV"""

    def __init__(self, model, training_dataset, validation_dataset, output_dir):
        """
        Parameters
        ----------
        model: estimator object
            The instantiated model class
        training_dataset: MultiInputDataset object
            The multi-input training dataset.
        validation_dataset: MultiInputDataset object
            The multi-input validation dataset.
        output_dir: str
            The output directory where the results of the hyperparameter search will be saved.
        """
        self.model = clone(model)
        self.X, self.y, self.predefined_split_inds = self._create_predefined_split(training_dataset, validation_dataset)

        self.output_dir = os.path.join(output_dir, 'hyperparam_opt')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.best_hyperparams = None

    def _create_predefined_split(self, training_set, validation_set):
        """Concatenate the training and validation sets and create indices so that a PredefinedSplit can be defined,
        allowing the use of TuneSearchCV later on."""
        new_dataset = training_set.concat_datasets(validation_set)
        new_dataset.concat_features()
        predefined_split_inds = []
        for i in range(new_dataset.y.shape[0]):
            if i < training_set.y.shape[0]:
                predefined_split_inds.append(-1)
            else:
                predefined_split_inds.append(0)
        return new_dataset.X, new_dataset.y, predefined_split_inds

    def search(self, n_configurations, hyperparam_space, main_metric, metric_mode, n_jobs=8, random_seed=None):
        """
        Tune the hyperparameters.

        Parameters
        ----------
        n_configurations: int
            Number of hyperparameter configurations to test.
        hyperparam_space: dict
            The search space.
        main_metric:
            The main scoring metric (used to determine the best configuration).
        metric_mode:
            Whether to maximize ('max') or minimize ('min') the main scoring metric.
        n_jobs:
            Number of jobs to run in parallel.
        random_seed:
            Seed to initialize the random number generator.

        Returns
        -------
        None

        """
        predefined_split = PredefinedSplit(test_fold=self.predefined_split_inds)
        tune_search = TuneSearchCV(estimator=self.model, param_distributions=hyperparam_space,
                                   early_stopping=False, n_trials=n_configurations, scoring=main_metric,
                                   search_optimization='bayesian', refit=False, cv=predefined_split,
                                   verbose=2, n_jobs=n_jobs, local_dir=self.output_dir,
                                   mode=metric_mode, random_state=random_seed)
        tune_search.fit(self.X, self.y)
        results = pd.DataFrame(data=tune_search.cv_results_)
        results.to_csv(os.path.join(self.output_dir, 'ray_tune_results.csv'), index=False)
        self.best_hyperparams = tune_search.best_params_
        # ray.shutdown()