import argparse
import sys

from src.preprocessing.preprocessing import OmicsDatasetPreprocessor

sys.setrecursionlimit(1000000)


def create_omics_files(expr_subnetwork_type, mut_subnetwork_type, cnv_subnetwork_type):
	"""
	Preprocess omics data for ALMANAC cell lines and save to file.

	Parameters
	----------
	expr_subnetwork_type: str
		The type of expression feature-encoding subnetwork that will be used (determines which preprocessing steps
		will be applied).
	mut_subnetwork_type: str
		The type of mutation feature-encoding subnetwork that will be used (determines which preprocessing steps
		will be applied).
	cnv_subnetwork_type: str
		The type of copy number variation (CNV) feature-encoding subnetwork that will be used (determines which
		preprocessing steps will be applied).
	"""
	id_col = 'CELLNAME'
	if expr_subnetwork_type is not None:
		if 'protein coding' in expr_subnetwork_type or 'WGCNA' in expr_subnetwork_type or 'UMAP' in expr_subnetwork_type:
			dataset_file = '../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_prot_coding.csv'
		else:
			dataset_file = '../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_all.csv'
		omics_preprocessor = OmicsDatasetPreprocessor(
			dataset_filepath=dataset_file,
			id_col='CELLNAME')

		subnetwork_type_to_output_names = {'expr (landmark)': 'rnaseq_fpkm_landmark_minmaxscaled',
									  'expr (NCG)': 'rnaseq_fpkm_ncg_minmaxscaled',
									  'expr (DGI)': 'rnaseq_fpkm_dgi_minmaxscaled',
									  'expr (DGI + NCG)': 'rnaseq_fpkm_dgi_ncg_minmaxscaled',
									  'expr (COSMIC)': 'rnaseq_fpkm_cosmic_minmaxscaled',
									  'expr (DGI + landmark)': 'rnaseq_fpkm_dgi_landmark_minmaxscaled',
									  'expr (UMAP)': 'umap',
									  'expr (WGCNA)': 'merged_rnaseq_fpkm_prot_coding',
									  'expr (protein coding, clustering order 1D CNN)': 'rnaseq_fpkm_prot_coding_1dconv_clustering_order',
									  'expr (protein coding, chromosome position order 1D CNN)': 'rnaseq_fpkm_prot_coding_1dconv_chromosome_order',
									  'expr (protein coding, chromosome position order 2D CNN)': 'rnaseq_fpkm_prot_coding_2dconv_chromosome_order',
									  'expr (protein coding, clustering order 2D CNN)': 'rnaseq_fpkm_prot_coding_2dconv_clustering_order',
									  'expr (protein coding)': 'rnaseq_fpkm_prot_coding_minmaxscaled'}

		if 'clustering' in expr_subnetwork_type:
			omics_preprocessor.get_clustering_gene_order(output_filepath='../data/expr_conv/clustering_order_minmaxscaler_complete_correlation.pkl')
			omics_preprocessor.reorder_genes(gene_order_filepath='../data/expr_conv/clustering_order_minmaxscaler_complete_correlation.pkl')
		elif 'chromosome' in expr_subnetwork_type:
			omics_preprocessor.get_chromosome_gene_order(output_filepath='../data/expr_conv/rnaseq_prot_coding_chromosome_position.pkl')
			omics_preprocessor.reorder_genes(gene_order_filepath='../data/expr_conv/rnaseq_prot_coding_chromosome_position.pkl')
		elif expr_subnetwork_type == 'expr (DGI)':
			omics_preprocessor.filter_genes(use_targets=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_dgi_genes.csv')
		elif expr_subnetwork_type == 'expr (landmark)':
			omics_preprocessor.filter_genes(use_landmark=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_lincs_landmark.csv')
		elif expr_subnetwork_type == 'expr (NCG)':
			omics_preprocessor.filter_genes(use_ncg=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_ncg_genes.csv')
		elif expr_subnetwork_type == 'expr (COSMIC)':
			omics_preprocessor.filter_genes(use_cosmic=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_cosmic_genes.csv')
		elif expr_subnetwork_type == 'expr (DGI + NCG)':
			omics_preprocessor.filter_genes(use_targets=True, use_ncg=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_dgi_ncg_genes.csv')
		elif expr_subnetwork_type == 'expr (DGI + landmark)':
			omics_preprocessor.filter_genes(use_targets=True, use_landmark=True)
			omics_preprocessor.save_full_dataset(
				output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_dgi_landmark_genes.csv')

		omics_preprocessor.merge(
			dataset_to_merge_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
		omics_preprocessor.split('../data/splits/train_val_test_groups_split_inds_12321.pkl')

		if expr_subnetwork_type == 'expr (UMAP)':
			omics_preprocessor.preprocess_split_datasets(scaler='MinMaxScaler',
														 umap_embeddings=True,
														 umap_n_components=50,
														 umap_n_neighbors=20)
		elif expr_subnetwork_type == 'expr (protein coding, clustering order 1D CNN)' or expr_subnetwork_type == 'expr (protein coding, chromosome position order 1D CNN)':
			omics_preprocessor.preprocess_split_datasets(scaler='MinMaxScaler', reshape_conv1d=True)
		elif expr_subnetwork_type == 'expr (protein coding, clustering order 2D CNN)' or expr_subnetwork_type == 'expr (protein coding, chromosome position order 2D CNN)':
			omics_preprocessor.preprocess_split_datasets(scaler='MinMaxScaler', reshape_conv2d=True, conv_2d_shape=138)
		elif expr_subnetwork_type == 'expr (WGCNA)':
			omics_preprocessor.save_split_datasets(output_dir='../data/nci_almanac_preprocessed/omics/split',
												   output_name=subnetwork_type_to_output_names[expr_subnetwork_type],
												   output_format='.csv.gz',
												   drop_id=False) # not scaling here, just going to save merged and split dataset to pass on to the R script
		else: # for the remaining expr options, just scale the data
			omics_preprocessor.preprocess_split_datasets(scaler='MinMaxScaler')

		omics_preprocessor.save_split_datasets(output_dir='../data/nci_almanac_preprocessed/omics/split',
											   output_name=subnetwork_type_to_output_names[expr_subnetwork_type],
											   output_format='.npy',
											   drop_id=True)

	if mut_subnetwork_type is not None:
		if mut_subnetwork_type == 'mut (DGI, gene-level)':
			mut_preprocessor = OmicsDatasetPreprocessor(dataset_filepath='../data/nci_almanac_preprocessed/omics/unmerged/mut_gene_level_binarized.csv',
			                                            id_col=id_col)
			mut_preprocessor.filter_genes(use_targets=True)
			mut_preprocessor.save_full_dataset(output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/mut_gene_level_binarized_dgi_genes.csv')
			output_name = 'merged_mut_gene_level_binarized_dgi_genes'
		elif mut_subnetwork_type == 'mut (pathway-level)':
			mut_preprocessor = OmicsDatasetPreprocessor(dataset_filepath='../data/nci_almanac_preprocessed/omics/unmerged/mut_pathway_level_binarized.csv',
			                                            id_col=id_col)
			output_name = 'merged_mut_pathway_level_binarized'
		mut_preprocessor.merge(
			dataset_to_merge_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
		mut_preprocessor.split('../data/splits/train_val_test_groups_split_inds_12321.pkl')
		mut_preprocessor.save_split_datasets(output_dir='../data/nci_almanac_preprocessed/omics/split',
		                                     output_name=output_name,
		                                     output_format='.npy',
		                                     drop_id=True)

	if cnv_subnetwork_type is not None:
		if cnv_subnetwork_type == 'cnv (DGI)':
			cnv_preprocessor = OmicsDatasetPreprocessor(dataset_filepath='../data/nci_almanac_preprocessed/omics/unmerged/cnvs_gistic_prot_coding.csv',
			                                            id_col=id_col)
			cnv_preprocessor.filter_genes(use_targets=True, use_aliases=False) # use_aliases = False because I didn't use them originally
			cnv_preprocessor.save_full_dataset(output_filepath='../data/nci_almanac_preprocessed/omics/unmerged/cnvs_gistic_dgi_genes.csv')
			cnv_preprocessor.merge(dataset_to_merge_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
			cnv_preprocessor.split('../data/splits/train_val_test_groups_split_inds_12321.pkl')
			cnv_preprocessor.save_split_datasets(output_dir='../data/nci_almanac_preprocessed/omics/split',
			                                     output_name='merged_cnvs_gistic_dgi_genes',
			                                     output_format='.npy',
			                                     drop_id=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess omics datasets and save to file')
	parser.add_argument('-e',
	                    '--expr-subnetwork-type',
	                    type=str,
	                    help='The type of expression feature-encoding subnetwork that will be used')
	parser.add_argument('-m',
	                    '--mut-subnetwork-type',
	                    type=str,
	                    help='The type of mutation feature-encoding subnetwork that will be used')
	parser.add_argument('-c',
	                    '--cnv-subnetwork-type',
	                    type=str,
	                    help='The type of copy number variation (CNV) feature-encoding subnetwork that will be used')
	args = vars(parser.parse_args())
	print(args)
	create_omics_files(**args)
