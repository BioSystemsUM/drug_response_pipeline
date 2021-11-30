import dill as pickle
from ray import tune
from skopt.space import Real, Integer, Categorical

# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [1],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (landmark, dense) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_bliss',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 128,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_landmark_drug_dense_1.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)


# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (full, dense) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_bliss',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 128,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_full_drug_dense_test.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'selu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'batchnorm': tune.choice([True, False]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}

# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (full, dense) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_bliss',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MaxAbsScaler',
#             'expr_transform': None,
#             }
# with open('settings_files/expr_full_maxabsscaler_drug_dense.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (landmark, dense) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_bliss',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MaxAbsScaler',
#             'expr_transform': None,
#             }
# with open('settings_files/expr_landmark_maxabsscaler_drug_dense.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

# #### Mol2vec config ####
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (full, dense, No Scaling) + drug (Mol2vec, dense) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_full_drug_mol2vec.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (landmark, dense, No Scaling) + drug (Mol2vec, dense) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_landmark_drug_mol2vec.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

# #### TextCNN config ####
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# print(str(char_dict))
# print(seq_length)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [7],
#             'model_type': 'expr_drug_textcnn_model',
#             'model_description': 'expr (landmark, dense, MinMaxScaler) + drug (TextCNN) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             }
# with open('settings_files/expr_landmark_minmaxscaler_drug_textcnn.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# settings = {'opt_hyperparams_path': 'settings_files/expr_full_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr_drug_textcnn_model',
#             'model_description': 'expr (full, dense, MinMaxScaler) + drug (TextCNN) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             }
# with open('settings_files/expr_full_minmaxscaler_drug_textcnn.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

#### WGCNA config ####
# config = {'expr_hlayers_sizes': tune.choice(['[128]', '[64]', '[32]', '[16]', '[8]',
#                                              '[128, 64]', '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[128, 64, 32]', '[64, 32, 16]', '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                                   '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                                   '[1024, 1024]', '[512, 512]', '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[1024, 1024, 1024]', '[512, 512, 512]', '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_wgcna_drug_ecfp4_1024_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_wgcna_drug_ecfp4_1024_hyperparam_space.pkl',
#             'gpu_to_use': [7],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (WGCNA, dense, No Scaling) + drug (ECFP4 1024, dense) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/full_dataset_module_eigengenes.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/train_module_eigengenes.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/val_module_eigengenes.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/test_module_eigengenes.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_wgcna_drug_ecfp4_1024.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]',
#                                              '[64, 32]', '[32, 16]', '[32, 8]',
#                                              '[64, 32, 16]', '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                                   '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                                   '[1024, 1024]', '[512, 512]', '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[1024, 1024, 1024]', '[512, 512, 512]', '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_wgcna_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_wgcna_drug_ecfp4_1024_hyperparam_space.pkl',
#             'gpu_to_use': [6],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (WGCNA, dense, No Scaling) + drug (Mol2vec, dense) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/mol2vec/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/full_dataset_module_eigengenes.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/train_module_eigengenes.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/val_module_eigengenes.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/test_module_eigengenes.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_wgcna_drug_mol2vec.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]',
#                                              '[64, 32]', '[32, 16]', '[32, 8]',
#                                              '[64, 32, 16]', '[32, 16, 8]']),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                              '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'predictor_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                                   '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[128, 64, 32]',
#                                                   '[1024, 1024]', '[512, 512]', '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[1024, 1024, 1024]', '[512, 512, 512]', '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_wgcna_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_wgcna_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [2],
#             'model_type': 'expr_drug_textcnn_model',
#             'model_description': 'expr (WGCNA, dense, No Scaling) + drug (TextCNN, dense) (Loewe)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles_without_MDA-MB-468.csv',
#             'output_col': 'synergy_loewe',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/textcnn/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/full_dataset_module_eigengenes.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/train_module_eigengenes.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/val_module_eigengenes.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/wgcna/test_module_eigengenes.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/split_inds_seed12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             }
# with open('settings_files/expr_wgcna_drug_textcnn.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

## Expr 1DConv
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_1dconv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_1dconv_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr1dconv_drug_dense_model',
#             'model_description': 'expr (full, 1D Conv) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_1dconv_minmaxscaler_drug_dense_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_full_1dconv_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_1dconv_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr1dconv_drug_textcnn_model',
#             'model_description': 'expr (full, 1D Conv) + drug (TextCNN)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_1dconv_minmaxscaler_drug_textcnn_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_full_1dconv_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_1dconv_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr1dconv_drug_dense_model',
#             'model_description': 'expr (full, 1D Conv) + drug (Mol2vec, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_1dconv_minmaxscaler_drug_mol2vec_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

# # Expr Conv 1D Landmark genes:
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_1dconv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_1dconv_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr1dconv_drug_dense_model',
#             'model_description': 'expr (landmark, 1D Conv) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_1dconv_minmaxscaler_drug_dense_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_landmark_1dconv_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_1dconv_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr1dconv_drug_textcnn_model',
#             'model_description': 'expr (landmark, 1D Conv) + drug (TextCNN)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_1dconv_minmaxscaler_drug_textcnn_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]',
#                                            '[16, 16]', '[32, 32]', '[64, 64]',
#                                            '[16, 32]', '[32, 64]']),
#           'expr_kernel_sizes': tune.choice(['[3, 3]', '[5, 5]', '[10, 10]', '[20, 20]']),
#           'expr_pool_size': tune.choice([5, 10]),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_landmark_1dconv_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_1dconv_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr1dconv_drug_dense_model',
#             'model_description': 'expr (landmark, 1D Conv) + drug (Mol2vec, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_1dconv_minmaxscaler_drug_mol2vec_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)


# Expr 2DConv
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': tune.choice([(3, 3), (5, 5), (7, 7)]),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_2dconv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_2dconv_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr2dconv_drug_dense_model',
#             'model_description': 'expr (full, 2D Conv) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d':False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 138,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_2dconv_minmaxscaler_drug_dense_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': tune.choice([(3, 3), (5, 5), (7, 7)]),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_full_2dconv_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_2dconv_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr2dconv_drug_textcnn_model',
#             'model_description': 'expr (full, 2D Conv) + drug (TextCNN)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 138,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_2dconv_minmaxscaler_drug_textcnn_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': tune.choice([(3, 3), (5, 5), (7, 7)]),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_full_2dconv_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_full_2dconv_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [6, 7],
#             'model_type': 'expr2dconv_drug_dense_model',
#             'model_description': 'expr (full, 2D Conv) + drug (Mol2vec, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 138,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset.pkl'
#             }
# with open('settings_files/expr_full_2dconv_minmaxscaler_drug_mol2vec_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)

# # Expr Conv 2D Landmark genes:
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': (3, 3),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_2dconv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_2dconv_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr2dconv_drug_dense_model',
#             'model_description': 'expr (landmark, 2D Conv) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 32,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_2dconv_minmaxscaler_drug_dense_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': (3, 3),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_landmark_2dconv_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_2dconv_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr2dconv_drug_textcnn_model',
#             'model_description': 'expr (landmark, 2D Conv) + drug (TextCNN)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 32,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_2dconv_minmaxscaler_drug_textcnn_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size1': (3, 3),
#           'expr_kernel_size_rest': (3, 3),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_landmark_2dconv_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_landmark_2dconv_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [2, 3],
#             'model_type': 'expr2dconv_drug_dense_model',
#             'model_description': 'expr (landmark, 2D Conv) + drug (Mol2vec, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_lincs_landmark.csv.gz',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': 'MinMaxScaler',
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': True,
#             'expr_conv_2d_shape': 32,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': '../data/nci_almanac_preprocessed/omics/expr_conv/gene_order_minmaxscaler_complete_correlation_notmergeddataset_landmark.pkl'
#             }
# with open('settings_files/expr_landmark_2dconv_minmaxscaler_drug_mol2vec_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)


# # UMAP
# config = {'expr_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]', '[8]', '[4]',
#                                              '[64, 32]', '[32, 16]', '[16, 8]', '[8, 4]',
#                                              '[64, 32, 16]', '[32, 16, 8]', '[16, 8, 4]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                                   '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                                   '[1024, 1024]', '[512, 512]', '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[1024, 1024, 1024]', '[512, 512, 512]', '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_umap_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
# #
# settings = {'opt_hyperparams_path': 'settings_files/expr_umap_drug_dense_hyperparam_space.pkl',
#             'gpu_to_use': [0, 1],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (UMAP) + drug (ECFP4, 1024, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_full.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_train.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_val.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_test.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             'expr_reshape_conv1d':False,
#             'expr_reshape_conv2d': False,
#             'expr_conv_2d_shape': None,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': None
#             }
# with open('settings_files/expr_umap_drug_dense_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
# config = {'expr_hlayers_sizes': tune.choice(['[32]', '[16]',
#                                              '[32, 16]', '[16, 8]',
#                                              '[32, 16, 8]']),
#           'drug_dropout': tune.choice([0.1, 0.2, 0.25]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', '[1, 2, 3, 4, 5, 8, 10, 15]']),
#           'drug_n_embedding': tune.choice([32, 75]),
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
#                                             '[50, 50, 50, 50, 50, 50, 50, 75]']),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'predictor_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                                   '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                                   '[1024, 1024]', '[512, 512]', '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[1024, 1024, 1024]', '[512, 512, 512]', '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_umap_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_umap_drug_textcnn_hyperparam_space.pkl',
#             'gpu_to_use': [0, 1],
#             'model_type': 'expr_drug_textcnn_model',
#             'model_description': 'expr (UMAP) + drug (TextCNN)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/TextCNN_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_full.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_train.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_val.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_test.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': False,
#             'expr_conv_2d_shape': None,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': None
#             }
# with open('settings_files/expr_umap_drug_textcnn_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[32]', '[16]',
#                                              '[32, 16]', '[32, 8]',
#                                              '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]',
#                                              '[256, 128]', '[128, 64]',
#                                              '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]', '[32]',
#                                                   '[256, 128]', '[128, 64]', '[64, 32]',
#                                                   '[256, 128, 64]', '[128, 64, 32]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[32, 32]',
#                                                   '[256, 256, 256]',
#                                                   '[128, 128, 128]', '[64, 64, 64]', '[32, 32, 32]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_umap_drug_mol2vec_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# settings = {'opt_hyperparams_path': 'settings_files/expr_umap_drug_mol2vec_hyperparam_space.pkl',
#             'gpu_to_use': [0, 1],
#             'model_type': 'expr_drug_dense_model',
#             'model_description': 'expr (UMAP) + drug (Mol2vec, dense)',
#             'response_dataset': '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
#             'output_col': 'COMBOSCORE',
#             'drugA_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugA.csv.gz',
#             'drugB_file': '../data/nci_almanac_preprocessed/drugs/Mol2VecFeaturizer_drugB.csv.gz',
#             'rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_full.csv',
#             'train_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_train.csv',
#             'val_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_val.csv',
#             'test_rnaseq_file': '../data/nci_almanac_preprocessed/omics/umap/umap_test.csv',
#             'split_type': 'TrainValTestSplit',
#             'split_inds': '../data/splits/train_val_test_groups_split_inds_12321.pkl',
#             'epochs': 500,
#             'batch_size': 64,
#             'main_metric': 'val_mean_squared_error',
#             'main_metric_mode': 'min',
#             'evaluation_output': '../results/model_evaluation_results_cellminercdb.csv',
#             'train_final_model': True,
#             'expr_variance_threshold': False,
#             'expr_scaler': None,
#             'expr_transform': None,
#             'expr_reshape_conv1d': False,
#             'expr_reshape_conv2d': False,
#             'expr_conv_2d_shape': None,
#             'expr_use_deepinsight': False,
#             'gene_order_filepath': None
#             }
# with open('settings_files/expr_umap_drug_mol2vec_ccdb.yml', 'w') as outfile:
# 	yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)


# ## Mut + CNVs + Expr (prot_coding) + Drugs, all dense, ECFP4
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'mut_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_mut_genelevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_mut_pathwaylevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# ## Mut + CNVs + drugs
# config = {'mut_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/mut_genelevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/mut_pathwaylevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# ## Mut + CNVs + Expr (landmark) + Drugs
# config = {'mut_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_mut_genelevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]',
#                                             '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_mut_pathwaylevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# # Expr 2DConv - redo
# config = {'expr_num_filters': tune.choice(['[16]', '[32]', '[64]', '[128]',
#                                            '[16, 32]', '[32, 64]', '[64, 128]',
#                                            '[16, 32, 64]', '[32, 64, 128]']),
#           'expr_kernel_size': tune.choice([(3, 3), (5, 5)]),
#           'expr_pool_size': (2, 2),
#           'expr_batchnorm': tune.choice([True, False]),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_2dconv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

## Mut + CNVs + Expr (prot_coding) + Drugs, all dense, ECFP4
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                             '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]',
#                                             '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_mut_genelevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'mut_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]',
#                                             '[512, 256, 128]', '[256, 128, 64]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                             '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]',
#                                             '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_mut_pathwaylevel_cnv_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# Expr Densenet
# config = {'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_2dconv__densenet_drug_dense_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[32]', '[16]', '[8]',
#                                              '[32, 16]', '[16, 8]',
#                                              '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}

# with open('settings_files/onehot_cells_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                              '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                              '[512, 256]', '[256, 128]',
#                                              '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                              '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]', '[8]',
#                                              '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[64, 32, 16]', '[32, 16, 8]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/onehot_drugs_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[32]', '[16]', '[8]',
#                                              '[32, 16]', '[16, 8]',
#                                              '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]', '[8]',
#                                              '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[64, 32, 16]', '[32, 16, 8]']),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/onehot_cells_onehot_drugs_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                             '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                             '[512, 256]', '[256, 128]',
#                                             '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                             '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.25, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]', '[8]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]', '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_cosmic_drug_ecpf4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]', '[32]', '[16]', '[8]',
#                                              '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]', '[32, 16, 8]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_target_genes_drug_ecpf4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_ecpf4_hyperparam_space_redo.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_target_genes_landmark_drug_ecpf4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_ecpf4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                             '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                             '[512, 256]', '[256, 128]',
#                                             '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                             '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_gat_layers': tune.choice(['[8, 8]', '[16, 16]', '[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[8, 8, 8]', '[16, 16, 16]', '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]']),
#           'drug_num_attention_heads': tune.choice([4, 6, 8]),
#           'drug_concat_heads': tune.choice([True, False]),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
#
# with open('settings_files/expr_full_drug_gat_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[8192]', '[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]',
#                                             '[8192, 4096]', '[4096, 2048]', '[2048, 1024]', '[1024, 512]',
#                                             '[512, 256]', '[256, 128]',
#                                             '[8192, 4096, 2048]', '[4096, 2048, 1024]', '[2048, 1024, 512]',
#                                             '[1024, 512, 256]', '[512, 256, 128]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]', '[256, 256]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]', '[256, 256, 256]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]', '[256, 256, 256, 256]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[4096]', '[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[4096, 2048]', '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[4096, 2048, 1024]', '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[4096, 4096]', '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[4096, 4096, 4096]', '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_full_drug_gcn_hyperparam_space_2.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_ncg_drug_ecpf4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]', '[256, 256]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]', '[256, 256, 256]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]', '[256, 256, 256, 256]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                     '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                     '[256, 128]', '[128, 64]',
#                                                     '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                     '[512, 256, 128]', '[256, 128, 64]',
#                                                     '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                     '[256, 256]', '[128, 128]', '[64, 64]',
#                                                     '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                     '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                     '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]', '[256, 256]',
#                                 '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]', '[256, 256, 256]',
#                                 '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]', '[256, 256, 256, 256]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_full_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_full_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_gat_layers': tune.choice(['[8, 8]', '[16, 16]', '[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[8, 8, 8]', '[16, 16, 16]', '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]']),
#           'drug_num_attention_heads': tune.choice([4, 6, 8]),
#           'drug_concat_heads': tune.choice([True, False]),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_full_drug_gat_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_full_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# print(str(char_dict))
# print(seq_length)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
#                                             '[1, 2, 3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10]',
#                                             '[3, 4, 5, 7]',
#                                             '[3, 4, 5]',
#                                             '[3, 5, 7]']),
#           'drug_n_embedding': tune.choice([32, 64, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]', # DeepChem default
#                                             '[32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]',
#                                             '[128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_targets_full_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_mut_cnv_targets_full_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                             '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                             '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_mut_pathway_cnv_targets_full_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/mut_cnv_targets_full_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                             '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                             '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/mut_pathway_cnv_targets_full_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# print(str(char_dict))
# print(seq_length)
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
#                                             '[1, 2, 3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10]',
#                                             '[3, 4, 5, 7]',
#                                             '[3, 4, 5]',
#                                             '[3, 5, 7]']),
#           'drug_n_embedding': tune.choice([32, 64, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]', # DeepChem default
#                                             '[32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]',
#                                             '[128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_targets_full_landmark_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_targets_full_landmark_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]', '[256, 256]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]', '[256, 256, 256]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]',
#                                           '[256, 256, 256, 256]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_targets_full_landmark_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
#                                             # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
#                                             '[1, 2, 3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10]',
#                                             '[3, 4, 5, 7]',
#                                             '[3, 4, 5]',
#                                             '[3, 5, 7]']),
#           'drug_n_embedding': tune.choice([32, 64, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters': tune.choice(
# 	          ['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',  # DeepChem default
# 	           '[32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]',
# 	           '[128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_targets_full_landmark_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]',
#                                              '[64, 32, 16]']),
#           'drug_gat_layers': tune.choice(['[8, 8]', '[16, 16]', '[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[8, 8, 8]', '[16, 16, 16]', '[32, 32, 32]', '[64, 64, 64]',
#                                           '[128, 128, 128]']),
#           'drug_num_attention_heads': tune.choice([4, 6, 8]),
#           'drug_concat_heads': tune.choice([True, False]),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_targets_full_landmark_drug_gat_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_gat_layers': tune.choice(['[8, 8]', '[16, 16]', '[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[8, 8, 8]', '[16, 16, 16]', '[32, 32, 32]', '[64, 64, 64]',
#                                           '[128, 128, 128]']),
#           'drug_num_attention_heads': tune.choice([4, 6, 8]),
#           'drug_concat_heads': tune.choice([True, False]),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_ncg_drug_gat_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutgene_cnv_ncg_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutpathway_cnv_ncg_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/mutgene_cnv_ncg_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/mutpathway_cnv_ncg_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.25, 0.5]),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutgene_cnv_ncg_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.25, 0.5]),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutpathway_cnv_ncg_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.25, 0.5]),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/mutgene_cnv_ncg_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]',
#                                           '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.25, 0.5]),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[2048, 1024]', '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[2048, 1024, 512]', '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/mutpathway_cnv_ncg_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)



# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_gcn_layers': tune.choice(['[32, 32]', '[64, 64]', '[128, 128]', '[256, 256]',
#                                 '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]', '[256, 256, 256]',
#                                 '[32, 32, 32, 32]', '[64, 64, 64, 64]', '[128, 128, 128, 128]', '[256, 256, 256, 256]']),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_gcn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_layeredfp_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_gat_layers': tune.choice(['[8, 8]', '[16, 16]', '[32, 32]', '[64, 64]', '[128, 128]',
#                                           '[8, 8, 8]', '[16, 16, 16]', '[32, 32, 32]', '[64, 64, 64]', '[128, 128, 128]']),
#           'drug_num_attention_heads': tune.choice([4, 6, 8]),
#           'drug_concat_heads': tune.choice([True, False]),
#           'drug_residual_connection': tune.choice([True, False]),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_gat_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]',
#                                              '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]',
#                                              '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_mtembeddings_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
#
# with open('char_dict_seq_len.pkl', 'rb') as f:
# 	char_dict, seq_length = pickle.load(f)
#
# print(str(char_dict))
# print(seq_length)
#
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'drug_kernel_sizes': tune.choice(['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]', # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
#                                             '[1, 2, 3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10, 15]',
#                                             '[3, 4, 5, 7, 10]',
#                                             '[3, 4, 5, 7]',
#                                             '[3, 4, 5]',
#                                             '[3, 5, 7]']),
#           'drug_n_embedding': tune.choice([32, 64, 75]),
#           'drug_char_dict': str(char_dict),  # because ray tune doesn't like dictionaries in config
#           'drug_seq_length': seq_length,
#           'drug_num_filters':  tune.choice(['[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]', # DeepChem default
#                                             '[32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]',
#                                             '[128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/expr_landmark_drug_textcnn_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

#
# # Mut gene level + CNV + Target genes Expr + ECFP4
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[512]', '[256]', '[128]', '[64]', '[32]', '[16]', '[8]',
#                                             '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]', '[16, 8]',
#                                             '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]', '[32, 16, 8]']),
#           'cnv_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutgene_cnv_targets_drug_ecfp4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# # Mut pathway level + CNV + Expr target genes + ECFP4
# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]',
#                                              '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]',
#                                              '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]']),
#           'mut_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'cnv_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
# with open('settings_files/expr_mutpathway_cnv_targets_drug_ecfp4_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'expr_hlayers_sizes': tune.choice(['[1024]', '[512]', '[256]', '[128]', '[64]', '[32]', '[16]',
#                                             '[1024, 512]', '[512, 256]', '[256, 128]', '[128, 64]', '[64, 32]', '[32, 16]',
#                                             '[1024, 512, 256]', '[512, 256, 128]', '[256, 128, 64]', '[128, 64, 32]', '[64, 32, 16]']),
#           'drug_hlayers_sizes': tune.choice(['[64]', '[32]', '[16]', '[8]',
#                                              '[64, 32]', '[32, 16]', '[16, 8]',
#                                              '[64, 32, 16]', '[32, 16, 8]']),
#           'predictor_hlayers_sizes': tune.choice(['[2048]', '[1024]', '[512]', '[256]', '[128]', '[64]',
#                                                   '[2048, 1024]', '[1024, 512]', '[512, 256]',
#                                                   '[256, 128]', '[128, 64]',
#                                                   '[2048, 1024, 512]', '[1024, 512, 256]',
#                                                   '[512, 256, 128]', '[256, 128, 64]',
#                                                   '[2048, 2048]', '[1024, 1024]', '[512, 512]',
#                                                   '[256, 256]', '[128, 128]', '[64, 64]',
#                                                   '[2048, 2048, 2048]', '[1024, 1024, 1024]',
#                                                   '[512, 512, 512]', '[256, 256, 256]', '[128, 128, 128]',
#                                                   '[64, 64, 64]']),
#           'hidden_dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
#           'hidden_activation': tune.choice(['relu', 'leakyrelu', 'prelu']),
#           'l2': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]),
#           'learn_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])}
#
# with open('settings_files/targets_onehot_drugs_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# ML
# config = {'n_estimators': [5, 10, 20, 50],
#           'min_samples_split': [2, 5, 10],
#           'max_depth': [20, 15, 10, 5],
#           'max_features': ['auto']
#           }
# with open('settings_files/test_ml.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'n_estimators': [100, 250, 500, 750, 1000],
#           'min_samples_split': [2, 3, 4, 5],
#           'max_depth': [None, 5, 10, 15, 20],
#           'min_samples_leaf': [1, 2, 5],
#           'max_features': ['auto', 'sqrt', 'log2'],
#           'n_jobs': [1]}
# with open('settings_files/ml/rf_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'alpha': Real(low=0.0001, high=1000, prior='log-uniform'),#(0.0001, 1000, 'log-uniform'),
#           'l1_ratio': Real(low=0.1, high=0.9, prior='uniform'), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#           'max_iter': [10000]
#           }
# with open('settings_files/ml/en_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)  # objective did not converge

# config = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
#           'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#           'max_iter': [100000]
#           }
# with open('settings_files/ml/en_hyperparam_space2.pkl', 'wb') as f:
# 	pickle.dump(config, f)
#
# config = {'tree_method': ['hist'],
#           'n_jobs': [1],
#           'n_estimators': [100, 250, 500, 750, 1000],
#           'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#           'max_depth': [3, 4, 5, 6],
#           "min_child_weight": [1, 5, 10],
#           "gamma": [0, 0.5, 1, 2],
#           "subsample": [0.6, 0.8, 1.0]}
# with open('settings_files/ml/xgb_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)


# config = {'C': (0.001, 1000, 'log-uniform'),
#           'gamma': (0.001, 1000, 'log-uniform'),
#           'epsilon': (0.001, 1000, 'log-uniform'),
#           'kernel': ['rbf', 'linear']
#           }
# with open('settings_files/ml/svr_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#           'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
#           'epsilon': [0.01, 0.1, 1],
#           'kernel': ['rbf', 'linear']
#           }
# with open('settings_files/ml/svr_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# # Use LinearSVR instead, as SVR doesn't scale well with this amount of data:
# config = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#           'epsilon': [0, 0.001, 0.01, 0.1, 1, 10],
#           'max_iter': [100000]
#           }
# with open('settings_files/ml/linearsvr_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# # Use Nystroem (to approximate RBF kernel) + LinearSVR instead, as SVR doesn't scale well:
# config = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#           'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
#           'epsilon': [0.01, 0.1, 1],
#           'kernel': ['rbf', 'linear']
#           }
# with open('settings_files/ml/svr_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)



# config = {'alpha': Real(low=0.0001, high=1000, prior='log-uniform'),#(0.0001, 1000, 'log-uniform'),
#           'l1_ratio': Real(low=0.1, high=0.9, prior='uniform'), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#           'max_iter': [100000]
#           }
# with open('settings_files/ml/en_continuous_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'n_estimators': Integer(low=100, high=1000),
#           'min_samples_split': Integer(low=2, high=5),
#           'max_depth': [None, 5, 10, 15, 20],
#           'min_samples_leaf': Integer(low=1, high=5),
#           'max_features': Categorical(['auto', 'sqrt', 'log2']),
#           'n_jobs': [1]}
# with open('settings_files/ml/rf_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'tree_method': ['hist'],
#           'n_jobs': [1],
#           'n_estimators': Integer(low=100, high=1000),
#           'learning_rate': Real(low=0.0001, high=0.1, prior='log-uniform'),
#           'max_depth': Integer(low=3, high=9),
#           "min_child_weight": Integer(low=1, high=5),
#           "gamma": Real(0, 2),
#           "subsample": Real(low=0.6, high=1.0)}
# with open('settings_files/ml/xgb_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

# config = {'C': Real(low=0.0001, high=10000, prior='log-uniform'),
#           'epsilon': Real(low=0.0001, high=10, prior='log-uniform'),
#           'max_iter': [100000]
#           }
# with open('settings_files/ml/linearsvr_hyperparam_space.pkl', 'wb') as f:
# 	pickle.dump(config, f)

config = {'C': Real(low=0.0001, high=1000, prior='log-uniform'),
          'epsilon': Real(low=0.0001, high=10, prior='log-uniform'),
          'loss': ['squared_epsilon_insensitive'],
          'dual': [False],
          'max_iter': [100000]
          }
with open('settings_files/ml/linearsvr_hyperparam_space2.pkl', 'wb') as f:
	pickle.dump(config, f)
