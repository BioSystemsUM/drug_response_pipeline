import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] ="2"
import argparse
from datetime import datetime

import dill as pickle
import yaml
import pickle

from src.dataset.dataset import MultiInputDataset
from src.model_selection.ml_model_selection import MLRayTuneSearch
from src.utils.utils import save_evaluation_results, get_ml_algorithm, evaluate_ml_model


def main(settings):
    # make output_dir
    output_dir = os.path.join('../results/', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir)

    # save settings to file in output_dir
    with open(os.path.join(output_dir, 'settings_used.yml'), 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

    # Load data
    print('Loading data')

    train_dataset = MultiInputDataset(settings['train_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                      output_col='COMBOSCORE')
    train_dataset.load_drugA(settings['train_drugA_filepath'])
    train_dataset.load_drugB(settings['train_drugB_filepath'])
    train_dataset.load_expr(settings['train_expr_filepath'])
    train_dataset.load_mut(settings['train_mut_filepath'])
    train_dataset.load_cnv(settings['train_cnv_filepath'])
    train_dataset.concat_features()

    val_dataset = MultiInputDataset(settings['val_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                          output_col='COMBOSCORE')
    val_dataset.load_drugA(settings['val_drugA_filepath'])
    val_dataset.load_drugB(settings['val_drugB_filepath'])
    val_dataset.load_expr(settings['val_expr_filepath'])
    val_dataset.load_mut(settings['val_mut_filepath'])
    val_dataset.load_cnv(settings['val_cnv_filepath'])
    val_dataset.concat_features()

    # Get model build function
    model = get_ml_algorithm(settings['model_type']) # get model class by name
    if 'hyperparams' in settings and settings['hyperparams'] is not None:
        model.set_params(**settings['hyperparams'])

    # Optimize hyperparameters
    if 'opt_hyperparams_path' in settings:
        print('Optimizing hyperparameters')
        with open(settings['opt_hyperparams_path'], 'rb') as f:
            opt_hyperparams = pickle.load(f)
        tuner = MLRayTuneSearch(model=model, training_dataset=train_dataset, validation_dataset=val_dataset,
                                output_dir=output_dir)

        tuner.search(n_configurations=50, hyperparam_space=opt_hyperparams, main_metric=settings['main_metric'],
                     metric_mode=settings['main_metric_mode'], n_jobs=1, random_seed=12321) # random_seed is ignored when using BOHB though
        best_hyperparams = tuner.best_hyperparams
    else:
        print('Not optimizing hyperparameters')
        best_hyperparams = settings['hyperparams']

    del val_dataset

    # Evaluate best model on test set
    print('Evaluating model')
    best_model = model.set_params(**best_hyperparams)
    best_model.fit(train_dataset.X, train_dataset.y)
    del train_dataset
    test_dataset = MultiInputDataset(settings['test_response_filepath'], id_cols=['CELLNAME', 'NSC1', 'NSC2'],
                                          output_col='COMBOSCORE')
    test_dataset.load_drugA(settings['test_drugA_filepath'])
    test_dataset.load_drugB(settings['test_drugB_filepath'])
    test_dataset.load_expr(settings['test_expr_filepath'])
    test_dataset.load_mut(settings['test_mut_filepath'])
    test_dataset.load_cnv(settings['test_cnv_filepath'])
    test_dataset.concat_features()
    evaluation_results = evaluate_ml_model(model, test_dataset, settings['scoring_metrics'])

    # Save evaluation results
    del test_dataset
    save_evaluation_results(evaluation_results, best_hyperparams, settings['model_description'], output_dir,
                            settings['evaluation_output'])

    # Save model
    model_path = os.path.join(output_dir, 'train_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate user-defined ML models')
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