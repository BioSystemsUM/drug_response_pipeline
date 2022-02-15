import os

import numpy as np
import ray
import tensorflow as tf
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.utils import pin_in_object_store, get_pinned_object
from ray.tune.suggest import bohb
from src.dataset.data_generator import DataGenerator


class KerasRayTuneSearch(object):
    """Tune hyperparameters using the Ray Tune implementation of Bayesian Optimization Hyperband (BOHB).
    BOHB uses Bayesian optimization to optimize the hyperparameters and Hyperband to early-stop bad trials."""

    def __init__(self, model_build_function, training_dataset, validation_dataset, epochs, n_gpus, output_dir,
                 batch_size):
        """
        Parameters
        ----------
        model_build_function: callable
            Function used to build the model.
        training_dataset: MultiInputDataset or DataGenerator object
            The multi-input training dataset.
        validation_dataset: MultiInputDataset object
            The multi-input validation dataset.
        epochs: int
            The number of epochs.
        n_gpus: int
            The number of GPUs to use.
        output_dir: str
            The output directory where the results of the hyperparameter search will be saved.
        batch_size: int
            The batch size.
        """
        ray.shutdown()
        ray.init(num_cpus=1, num_gpus=n_gpus, log_to_driver=False, _lru_evict=True)
        self.build_fn = model_build_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = os.path.join(output_dir, 'hyperparam_opt')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if isinstance(training_dataset, DataGenerator):
            self.training_generator = training_dataset
            self.use_data_generator = True
        else:
            self.X_train = pin_in_object_store(training_dataset.X_dict)
            self.y_train = pin_in_object_store(training_dataset.y)
            self.use_data_generator = False
        self.X_val = pin_in_object_store(validation_dataset.X_dict)
        self.y_val = pin_in_object_store(validation_dataset.y)

        # self.best_model = None
        self.best_hyperparams = None

    def _run_experiment(self, config):
        """Create a model with the provided hyperparameter configuration and train it."""
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.build_fn(**config)
        callbacks = [TuneReportCallback()]
        X_val = get_pinned_object(self.X_val)
        y_val = get_pinned_object(self.y_val)
        if self.use_data_generator:
            model.fit(self.training_generator,
                      validation_data=(X_val, y_val),
                      verbose=0,
                      epochs=self.epochs,
                      callbacks=callbacks,
                      workers=6, use_multiprocessing=False,
                      validation_batch_size=64)
        else:
            X_train = get_pinned_object(self.X_train)
            y_train = get_pinned_object(self.y_train)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      verbose=0,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      callbacks=callbacks,
                      validation_batch_size=64)

    def search(self, n_configurations, hyperparam_space, main_metric, metric_mode, n_cpus_per_trial, n_gpus_per_trial,
               random_seed):
        """
        Tune the hyperparameters.

        Parameters
        ----------
        n_configurations: int
            Number of hyperparameter configurations to test.
        hyperparam_space: dict
            The search space.
        main_metric: str
            The main scoring metric (used to determine the best configuration).
        metric_mode: str
            Whether to maximize ('max') or minimize ('min') the main scoring metric.
        n_cpus_per_trial: int
            Number of CPUs allowed per trial.
        n_gpus_per_trial: int
            Number of GPUs allowed per trial.
        random_seed: int
            Seed to initialize the random number generator.

        Returns
        -------
        results: DataFrame
            The results of the hyperparameter optimization run.
        """
        np.random.seed(random_seed)
        bohb_hyperband = ray.tune.schedulers.HyperBandForBOHB(time_attr='training_iteration',
                                                              max_t=100,
                                                              reduction_factor=4)
        bohb_search = bohb.TuneBOHB()
        analysis = ray.tune.run(self._run_experiment, verbose=1, config=hyperparam_space, num_samples=n_configurations,
                                search_alg=bohb_search, scheduler=bohb_hyperband, local_dir=self.output_dir,
                                resources_per_trial={"cpu": n_cpus_per_trial, "gpu": n_gpus_per_trial},
                                metric=main_metric,
                                mode=metric_mode,
                                max_failures=3,
                                queue_trials=True)
        ray.shutdown()

        self.best_hyperparams = analysis.best_config
        results = analysis.results_df
        results.to_csv(os.path.join(self.output_dir, 'ray_tune_results.csv'), index=False)

        return results
