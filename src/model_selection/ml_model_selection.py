import os

import pandas as pd
from sklearn import clone
from sklearn.model_selection import PredefinedSplit
from tune_sklearn import TuneSearchCV


class MLRayTuneSearch(object):

    def __init__(self, model, training_dataset, validation_dataset, output_dir):
        self.model = clone(model)
        self.X, self.y, self.predefined_split_inds = self._create_predefined_split(training_dataset, validation_dataset)

        self.output_dir = os.path.join(output_dir, 'hyperparam_opt')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.best_hyperparams = None

    def _create_predefined_split(self, training_set, validation_set):
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