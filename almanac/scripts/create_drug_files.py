import argparse

from src.preprocessing.preprocessing import DrugDatasetPreprocessor
from src.utils.utils import build_char_dict


def create_drug_files(featurization_type):
	"""
	Featurize ALMANAC drugs.

	Parameters
	----------
	featurization_type: str
		The featurization method that will be applied.
	"""
	drugA_preprocessor = DrugDatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		id_col='NSC1',
		smiles_col='SMILES_A')
	drugB_preprocessor = DrugDatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		id_col='NSC2',
		smiles_col='SMILES_B')

	# Split datasets before featurizing:
	drugA_preprocessor.split(
		split_inds_file='../data/splits/train_val_test_groups_split_inds_12321.pkl')
	drugB_preprocessor.split(
		split_inds_file='../data/splits/train_val_test_groups_split_inds_12321.pkl')

	# For TextCNN
	char_dict, seq_length = build_char_dict('../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
											smiles_cols=['SMILES_A', 'SMILES_B'],
											save=False)
	# For GCN and GAT
	max_n_atoms_A = drugA_preprocessor._get_max_number_of_atoms()
	max_n_atoms_B = drugB_preprocessor._get_max_number_of_atoms()
	max_n_atoms_whole_dataset = max(max_n_atoms_A, max_n_atoms_B)

	# Featurize drug A and drug B:
	featurizer_options = {'ECFP4': ('ECFPFeaturizer', {'radius': 2, 'length': 1024}, 'ECFP4_1024_drugA', 'ECFP4_1024_drugB'),
						  'ECFP6': ('ECFPFeaturizer', {'radius': 3, 'length': 1024}, 'ECFP6_1024_drugA', 'ECFP6_1024_drugB'),
						  'LayeredFP': ('LayeredFPFeaturizer', {'fp_size': 1024}, 'LayeredFPFeaturizer_drugA', 'LayeredFPFeaturizer_drugB'),
						  'GCN': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': True, 'max_num_atoms':max_n_atoms_whole_dataset}, 'GCN_drugA', 'GCN_drugB'),
						  'GAT': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': False, 'max_num_atoms':max_n_atoms_whole_dataset}, 'GAT_drugA', 'GAT_drugB'),
						  'MTE': ('MTEmbeddingsFeaturizer', {'embeddings_file':'../data/nci_almanac_preprocessed/drugs/mtembeddings_almanac_smiles.npz'}, 'MTEmbeddingsFeaturizer_drugA', 'MTEmbeddingsFeaturizer_drugB'),
						  'TextCNN': ('TextCNNFeaturizer', {'char_dict':char_dict, 'seq_length':seq_length}, 'TextCNNFeaturizer_drugA', 'TextCNNFeaturizer_drugB')}
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