import argparse
import os
from datetime import datetime

import dill as pickle
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model

from src.dataset.data_generator import DataGenerator
from src.dataset.dataset import MultiInputDataset
from src.model_selection.model_selection import KerasRayTuneSearch
from src.models import drug_pairs_build_functions
from src.utils.utils import save_evaluation_results, plot_keras_history, get_dataset_dims


def main(settings):
    """
    Train and evaluate deep learning models on data from the ALMANAC drug combination screen.

    Parameters
    ----------
    settings: dict
        Dictionary with user-defined settings and model configurations.

    """
    if 'gpu_to_use' in settings:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['gpu_to_use'])[1:-1]
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)

    strategy = tf.distribute.MirroredStrategy()  # to use multiple GPUs

    # make output_dir
    output_dir = os.path.join('../results/', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir)

    # save settings to file in output_dir
    with open(os.path.join(output_dir, 'settings_used.yml'), 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    # Load data
    print('Loading data')
    if settings['use_data_generators']:
        for key in ['train_drugA_filepath', 'train_drugB_filepath', 'train_expr_filepath', 'train_mut_filepath',
                    'train_cnv_filepath', 'train_drugA_atom_feat_filepath', 'train_drugB_atom_feat_filepath',
                    'train_drugA_adj_filepath', 'train_drugB_adj_filepath']:
            if settings[key] is not None:
                settings[key] = os.path.abspath(settings[key])
                # doing this because numpy load with mmap_mode in DataGenerator is not finding the files when using relative paths

        train_dataset = DataGenerator(y_filepath=settings['train_response_filepath'],
                                      output_col='COMBOSCORE',
                                      drugA_filepath=settings['train_drugA_filepath'],
                                      drugB_filepath=settings['train_drugB_filepath'],
                                      expr_filepath=settings['train_expr_filepath'],
                                      mut_filepath=settings['train_mut_filepath'],
                                      cnv_filepath=settings['train_cnv_filepath'],
                                      drugA_atom_feat_filepath=settings['train_drugA_atom_feat_filepath'],
                                      drugB_atom_feat_filepath=settings['train_drugB_atom_feat_filepath'],
                                      drugA_adj_filepath=settings['train_drugA_adj_filepath'],
                                      drugB_adj_filepath=settings['train_drugB_adj_filepath'],
                                      batch_size=settings['batch_size'],
                                      shuffle=True)
        dataset_dims = get_dataset_dims(train_dataset.X_filepaths, model_type=settings['model_type'])
    else:
        train_dataset = MultiInputDataset(settings['train_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                          output_col='COMBOSCORE')
        train_dataset.load_drugA(settings['train_drugA_filepath'])
        train_dataset.load_drugB(settings['train_drugB_filepath'])
        train_dataset.load_expr(settings['train_expr_filepath'])
        train_dataset.load_mut(settings['train_mut_filepath'])
        train_dataset.load_cnv(settings['train_cnv_filepath'])
        train_dataset.load_graph_data(nodes_file=settings['train_drugA_atom_feat_filepath'],
                                      adj_file=settings['train_drugA_adj_filepath'])
        train_dataset.load_graph_data(nodes_file=settings['train_drugB_atom_feat_filepath'],
                                      adj_file=settings['train_drugB_adj_filepath'])
        dataset_dims = train_dataset.get_dataset_dimensions(model_type=settings['model_type'])

    val_dataset = MultiInputDataset(settings['val_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                    output_col='COMBOSCORE')
    val_dataset.load_drugA(settings['val_drugA_filepath'])
    val_dataset.load_drugB(settings['val_drugB_filepath'])
    val_dataset.load_expr(settings['val_expr_filepath'])
    val_dataset.load_mut(settings['val_mut_filepath'])
    val_dataset.load_cnv(settings['val_cnv_filepath'])
    val_dataset.load_graph_data(nodes_file=settings['val_drugA_atom_feat_filepath'],
                                adj_file=settings['val_drugA_adj_filepath'])
    val_dataset.load_graph_data(nodes_file=settings['val_drugB_atom_feat_filepath'],
                                adj_file=settings['val_drugB_adj_filepath'])

    # Get model build function
    build_function = getattr(drug_pairs_build_functions, settings['model_type'])

    # Hyperparameter optimization using validation set
    if 'opt_hyperparams_path' in settings:
        print('Optimizing hyperparameters')
        with open(settings['opt_hyperparams_path'], 'rb') as f:
            opt_hyperparams = pickle.load(f)
        opt_hyperparams.update(dataset_dims)
        tuner = KerasRayTuneSearch(model_build_function=build_function,
                                   training_dataset=train_dataset,
                                   validation_dataset=val_dataset,
                                   epochs=settings['epochs'],
                                   batch_size=settings['batch_size'],
                                   n_gpus=len(settings['gpu_to_use']),
                                   output_dir=output_dir)
        tuner.search(n_configurations=50, hyperparam_space=opt_hyperparams, random_seed=12321,
                     main_metric=settings['main_metric'], metric_mode=settings['main_metric_mode'],
                     n_cpus_per_trial=1, n_gpus_per_trial=len(settings['gpu_to_use']))
        best_hyperparams = tuner.best_hyperparams
    else:
        print('Not optimizing hyperparameters')
        best_hyperparams = settings['hyperparams']
        best_hyperparams.update(dataset_dims)

    # Model evaluation on test set (validation set used for EarlyStopping)
    print('Evaluating model on test set')
    with strategy.scope():
        best_model = build_function(**best_hyperparams)
    # refitting best model so I can plot history
    if settings['use_data_generators']:
        history = best_model.fit(x=train_dataset, epochs=settings['epochs'], batch_size=settings['batch_size'],
                                 callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                                            CSVLogger(os.path.join(output_dir, 'training.log'))],
                                 validation_data=(val_dataset.X_dict, val_dataset.y), workers=6,
                                 use_multiprocessing=False, validation_batch_size=64)
        train_dataset.delete_memmaps()
    else:
        history = best_model.fit(train_dataset.X_dict, train_dataset.y, epochs=settings['epochs'],
                                 batch_size=settings['batch_size'],
                                 callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                                            CSVLogger(os.path.join(output_dir, 'training.log'))],
                                 validation_data=(
                                     val_dataset.X_dict, val_dataset.y), validation_batch_size=64)

    # best number of epochs
    best_n_epochs = np.argmin(history.history['val_loss']) + 1
    print('best n_epochs: %s' % best_n_epochs)

    del train_dataset
    del val_dataset
    test_dataset = MultiInputDataset(settings['test_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                     output_col='COMBOSCORE')
    test_dataset.load_drugA(settings['test_drugA_filepath'])
    test_dataset.load_drugB(settings['test_drugB_filepath'])
    test_dataset.load_expr(settings['test_expr_filepath'])
    test_dataset.load_mut(settings['test_mut_filepath'])
    test_dataset.load_cnv(settings['test_cnv_filepath'])
    test_dataset.load_graph_data(nodes_file=settings['test_drugA_atom_feat_filepath'],
                                 adj_file=settings['test_drugA_adj_filepath'])
    test_dataset.load_graph_data(nodes_file=settings['test_drugB_atom_feat_filepath'],
                                 adj_file=settings['test_drugB_adj_filepath'])

    evaluation_results = best_model.evaluate(x=test_dataset.X_dict, batch_size=64, y=test_dataset.y, return_dict=True)
    save_evaluation_results(evaluation_results, best_hyperparams, settings['model_description'], output_dir,
                            settings['evaluation_output'])
    del test_dataset
    # TODO: add option to save predictions (not saving them for all models because files are too big)

    try:
        best_model.save(os.path.join(output_dir, 'train_set_model'))
    except Exception as e:
        print(e)
        best_model.save(os.path.join(output_dir, 'train_set_model.h5'))

    plot_keras_history(history, output_dir)
    plot_model(best_model, to_file=os.path.join(output_dir, 'train_set_model.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate user-defined models')
    parser.add_argument('settings',
                        metavar='f',
                        type=str,
                        help='Path to the settings file (YAML format) that will be used to setup the model. '
                             'The file includes dataset options, fixed hyperparameter values, and hyperparameters to optimize')
    args = parser.parse_args()

    # Parse settings file
    with open(args.settings, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)  # loaded is a dict
        except yaml.YAMLError as exc:
            print(exc)

    main(loaded)
