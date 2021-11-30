import argparse

from src.preprocessing.preprocessing import DrugDatasetPreprocessor


def create_drug_files(featurization_type):
	drugA_preprocessor = DrugDatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		id_col='NSC1',
		smiles_col='SMILES_A')
	drugB_preprocessor = DrugDatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		id_col='NSC2',
		smiles_col='SMILES_B')

	# Split datasets before featurizing:
	drugA_preprocessor.split(
		split_inds_file='../data/splits/almanac/data/splits/train_val_test_groups_split_inds_12321.pkl')
	drugB_preprocessor.split(
		split_inds_file='../data/splits/almanac/data/splits/train_val_test_groups_split_inds_12321.pkl')

	# Featurize drug A and drug B:
	featurizer_options = {'ECFP4': ('ECFPFeaturizer', {'radius': 2, 'length': 1024}, 'ECFP4_1024_drugA', 'ECFP4_1024_drugB'),
	               'ECFP6': ('ECFPFeaturizer', {'radius': 3, 'length': 1024}, 'ECFP6_1024_drugA', 'ECFP6_1024_drugB'),
	               'LayeredFP': ('LayeredFPFeaturizer', {'fp_size': 1024}, 'LayeredFPFeaturizer_drugA', 'LayeredFPFeaturizer_drugB'),
	               'GCN': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': True}, 'GCN_drugA', 'GCN_drugB'),
	               'GAT': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': False}, 'GAT_drugA', 'GAT_drugB'),
	               'MTE': ('MTEmbeddingsFeaturizer', {}, 'MTEmbeddingsFeaturizer_drugA', 'MTEmbeddingsFeaturizer_drugB')}
	featurizer_opt = featurizer_options[featurization_type]
	drugA_preprocessor.featurize(featurizer_opt[0], featurizer_args=featurizer_opt[1],
	                             output_dir='../data/nci_almanac_preprocessed/drugs',
	                             output_prefix=featurizer_opt[2], featurize_split_datasets=True,
	                             featurize_full_dataset=False)  # features are automatically saved as .npy files
	drugB_preprocessor.featurize(featurizer_opt[0], featurizer_args=featurizer_opt[1],
	                             output_dir='../data/nci_almanac_preprocessed/drugs',
	                             output_prefix=featurizer_opt[3], featurize_split_datasets=True,
	                             featurize_full_dataset=False)  # features are automatically saved as .npy files


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Featurize ALMANAC drugs and save to file')
	parser.add_argument('-t',
	                    '--featurization-type',
	                    type=str,
	                    help='The type of featurization to use')
	args = vars(parser.parse_args())
	print(args)
	create_drug_files(**args)