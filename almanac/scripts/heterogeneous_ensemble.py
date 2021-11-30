import os

from spektral.layers import GCNConv, GlobalSumPool

from dataset import MultiInputDataset
from ensembles import VotingEnsemble
from utils import save_evaluation_results
import scoring_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def run_ensemble(model_description, output_file):
	print('loading test data')
	dgi_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	dgi_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	dgi_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')
	dgi_ecfp4_test_dataset.concat_features()

	# landmark_ecfp4_test_dataset
	landmark_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	landmark_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	landmark_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	landmark_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_landmark_minmaxscaled_test.npy')

	# cosmic_ecfp4_test_dataset
	cosmic_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	cosmic_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	cosmic_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	cosmic_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_cosmic_minmaxscaled_test.npy')

	# ncg_ecfp4_test_dataset
	ncg_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	ncg_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	ncg_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	ncg_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_ncg_minmaxscaled_test.npy')

	# dgi_landmark_ecfp4_test_dataset
	dgi_landmark_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_landmark_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	dgi_landmark_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	dgi_landmark_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_landmark_minmaxscaled_test.npy')

	# dgi_ncg_ecfp4_test_dataset
	dgi_ncg_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_ncg_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	dgi_ncg_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	dgi_ncg_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_ncg_minmaxscaled_test.npy')

	# dgi_gcn_test_dataset
	dgi_gcn_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_gcn_test_dataset.load_graph_data(nodes_file='../data/nci_almanac_preprocessed/drugs/GCN_drugA_nodes_test.npy', adj_file='../data/nci_almanac_preprocessed/drugs/GCN_drugA_adjmatrix_test.npy')
	dgi_gcn_test_dataset.load_graph_data(nodes_file='../data/nci_almanac_preprocessed/drugs/GCN_drugB_nodes_test.npy', adj_file='../data/nci_almanac_preprocessed/drugs/GCN_drugB_adjmatrix_test.npy')
	dgi_gcn_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')


	# dgi_layeredfp_test_dataset
	dgi_layeredfp_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_layeredfp_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/LayeredFPFeaturizer_drugA_test.npy')
	dgi_layeredfp_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/LayeredFPFeaturizer_drugB_test.npy')
	dgi_layeredfp_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')

	# dgi_mte_test_dataset
	dgi_mte_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_mte_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/MTEmbeddingsFeaturizer_drugA_test.npy')
	dgi_mte_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/MTEmbeddingsFeaturizer_drugB_test.npy')
	dgi_mte_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')

	dgi_mut_cnv_ecfp4_test_dataset = MultiInputDataset(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
		id_cols=['CELLNAME', 'NSC1', 'NSC2'],
		output_col='COMBOSCORE')
	dgi_mut_cnv_ecfp4_test_dataset.load_drugA('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugA_test.npy')
	dgi_mut_cnv_ecfp4_test_dataset.load_drugB('../data/nci_almanac_preprocessed/drugs/ECFP4_1024_drugB_test.npy')
	dgi_mut_cnv_ecfp4_test_dataset.load_expr(
		'../data/nci_almanac_preprocessed/omics/split/rnaseq_fpkm_targets_full_minmaxscaled_test.npy')
	dgi_mut_cnv_ecfp4_test_dataset.load_mut('../data/nci_almanac_preprocessed/omics/split/merged_mut_pathway_level_binarized_test.npy')
	dgi_mut_cnv_ecfp4_test_dataset.load_cnv('../data/nci_almanac_preprocessed/omics/split/merged_cnvs_gistic_target_genes_test.npy')


	paths_to_saved_models = ['../results/2021-07-06_12-19-03/train_set_model.h5',  # DL (expr (DGI) + drugs (ECFP4)
	                         '../results/2021-07-06_12-02-29/train_set_model.h5', # DL (expr (landmark) + drugs (ECFP4)
	                         '../results/2021-07-06_12-12-35/train_set_model.h5', # DL (expr (COSMIC) + drugs (ECFP4)
	                         '../results/2021-09-09_15-05-24/train_set_model', # DL (expr (target genes, dense, MinMaxScaler) + mut (pathway-level, target genes) + cnv (GISTIC, target genes) + drug (ECFP4, Dense))
	                         '../results/2021-07-13_20-03-54/train_set_model', # LayeredFP
	                         '../results/2021-07-13_15-43-50/train_set_model', # MTE
	                         '../results/2021-07-06_12-21-29/train_set_model.h5', # NCG
	                         '../results/2021-07-06_12-37-08/train_set_model.h5', # DGI + landmark
	                         '../results/2021-07-06_12-38-01/train_set_model.h5', # DGI + NCG
	                         '../results/2021-10-26_14-36-02/train_model.pkl',  # LGBM
	                         '../results/2021-09-27_12-58-08/train_model.pkl',  # XGBoost
	                         '../results/2021-09-27_23-01-20/train_model.pkl', # RF
	                         '../results/2021-07-14_22-20-49/train_set_model',  # GCN
	                         ]

	datasets = [dgi_ecfp4_test_dataset,
	            landmark_ecfp4_test_dataset,
	            cosmic_ecfp4_test_dataset,
	            dgi_mut_cnv_ecfp4_test_dataset,
	            dgi_layeredfp_test_dataset,
	            dgi_mte_test_dataset,
	            ncg_ecfp4_test_dataset,
	            dgi_landmark_ecfp4_test_dataset,
	            dgi_ncg_ecfp4_test_dataset,
	            dgi_ecfp4_test_dataset,
	            dgi_ecfp4_test_dataset,
	            dgi_ecfp4_test_dataset,
	            dgi_gcn_test_dataset,
	            ]
	custom_objects = []
	for model in paths_to_saved_models:
		if model == '../results/2021-07-14_22-20-49/train_set_model':
			custom_objects.append({'keras_r2_score': scoring_metrics.keras_r2_score,
			                       'keras_spearman': scoring_metrics.keras_spearman,
			                       'keras_pearson': scoring_metrics.keras_pearson,
			                       'GCNConv': GCNConv,
			                       'GlobalSumPool': GlobalSumPool})
		else:
			custom_objects.append({'keras_r2_score': scoring_metrics.keras_r2_score,
			                       'keras_spearman': scoring_metrics.keras_spearman,
			                       'keras_pearson': scoring_metrics.keras_pearson})
	print('creating ensemble')
	ensemble = VotingEnsemble(base_models=None, mode='regression')
	ensemble.load_models(paths_to_saved_models, custom_objects)
	results = ensemble.evaluate(datasets, metrics=['r2_score', 'mean_squared_error', 'pearson', 'spearman'])
	save_evaluation_results(results, hyperparams={},
	                        model_descr=model_description,
	                        model_dir='',
	                        output_filepath=output_file)


if __name__ == '__main__':
	run_ensemble(model_description='Voting Ensemble - Top 10 DL models +LGBM+XGBoost+RF',
	             output_file='../results/ensemble_evaluation_results.csv')