import os

import numpy as np
import ray
import tensorflow as tf
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.utils import pin_in_object_store, get_pinned_object
from ray.tune.suggest import bohb
from src.dataset.data_generator import DataGenerator


class KerasRayTuneSearch(object):

    def __init__(self, model_build_function, training_dataset, validation_dataset, epochs, n_gpus, output_dir,
                 batch_size):
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
        #wait_for_gpu(gpu_memory_limit=0.01) #TODO pip install gputil.
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
        np.random.seed(random_seed)
        bohb_hyperband = ray.tune.schedulers.HyperBandForBOHB(time_attr='training_iteration',
                                                              max_t=100,
                                                              reduction_factor=4)
        #bohb_search = ray.tune.suggest.bohb.TuneBOHB()
        bohb_search = bohb.TuneBOHB()
        analysis = ray.tune.run(self._run_experiment, verbose=1, config=hyperparam_space, num_samples=n_configurations,
                                search_alg=bohb_search, scheduler=bohb_hyperband, local_dir=self.output_dir,
                                resources_per_trial={"cpu": n_cpus_per_trial, "gpu": n_gpus_per_trial},
                                metric=main_metric,
                                mode=metric_mode,
                                max_failures=3,
                                queue_trials=True)
                                # queue_trials=True,
                                # raise_on_failed_trial=False)
        ray.shutdown()

        self.best_hyperparams = analysis.best_config
        # self.best_model = keras.models.load_model(os.path.join(analysis.best_logdir, 'model.h5'))
        results = analysis.results_df
        results.to_csv(os.path.join(self.output_dir, 'ray_tune_results.csv'), index=False)

        return results


