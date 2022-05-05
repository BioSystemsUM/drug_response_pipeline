import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pubchempy as pcp
from rdkit.Chem.PandasTools import LoadSDF

from src.preprocessing.standardization import SmilesStandardizer
from src.preprocessing.preprocessing import DatasetPreprocessor

sys.setrecursionlimit(1000000)


def preprocess_almanac_cellminercdb(output_path, remove_duplicate_triplets=False):
	"""
	Preprocess the ALMANAC drug response data from CellMinerCDB. Removes cell lines with missing data, gets SMILES
	strings for each drug and standardizes the compounds using the ChEMBL Structure Pipeline.

	Parameters
	----------
	output_path:
		Path to the file where the preprocessed response dataset will be saved.
	remove_duplicate_triplets: bool
		Whether to remove duplicate cell line-drugA-drugB triplets

	Returns
	-------
	None
	"""
	df = pd.read_table('../data/CellminerCDB/data_NCI60-DTP Almanac_act.txt').drop(['MOA', 'CLINICAL.STATUS'], axis=1)
	df.rename(columns=lambda x: x.split(':')[-1], inplace=True)
	df = df.melt(id_vars=['ID', 'NAME'], var_name='CELLNAME',
	             value_name='COMBOSCORE')  # convert from wide to long format
	df.dropna(axis=0,
	          inplace=True)  # this also removes the MDA-N cell line which was all NAs and is not part of the original ALMANAC dataset
	# The MDA-MB-468 cell line that's present in the ALMANAC dataset doesn't appear in the CellMinerCDB dataset

	# Check for duplicate cell line-drugA-drugB triplets and inverted duplicates (drug order is reversed -
	#  cell line-drugB-drugA) and use average values instead
	if remove_duplicate_triplets:
		unique_ids = df['ID'].unique().tolist()
		ids_to_groups_mapper = {}
		for id in unique_ids:
			inv_id = '_'.join(id.split('_')[::-1])
			if inv_id in ids_to_groups_mapper:
				ids_to_groups_mapper[id] = inv_id
			else:
				ids_to_groups_mapper[id] = id
		df['GROUP'] = df['ID'].apply(
			lambda x: ids_to_groups_mapper[x])  # GROUP = the drug combination independent of drug order
		df_duplicates = df[df.duplicated(subset=['GROUP', 'CELLNAME'],
		                                 keep=False)]  # 12828 triplets with the same cell line + drug combination, but with the reverse drug order
		#df_duplicates.to_csv('duplicate_triples_reversed_order_drugs.csv')
		df_duplicates_avg = df_duplicates.groupby(['GROUP', 'CELLNAME'],
		                                          as_index=False).mean()  # calculate average COMBOSCORE of the multiple cell line-treatment experiments
		df_duplicates_avg['ID'] = df_duplicates_avg['GROUP']
		df.drop_duplicates(subset=['GROUP', 'CELLNAME'], keep=False,
		                   inplace=True)  # 293537 rows; previously 306365 rows
		df = pd.concat([df, df_duplicates_avg]).reset_index(drop=True)  # 299951 rows

	# Get SMILES strings
	df[['NSC1', 'NSC2']] = df['ID'].str.split('_', 1, expand=True)
	sdf_df = LoadSDF('../data/ALMANAC/ComboCompoundSet.sdf', smilesName='smiles', isomericSmiles=False)
	# RDKit doesn't load NSCs '119875' and '266046' for some reason, and NSC '753082' is not in the SDF file
	# NSCs 761431 and 753082 are both Vemurafenib...
	# use pubchempy to get missing SMILES
	compound_names_df = pd.read_table('../data/ALMANAC/ComboCompoundNames_small.txt', header=None,
	                                  names=['nsc', 'name'])
	sdf_df = sdf_df[~sdf_df['NSC'].isin(['119875', '266046', '753082'])] # because if we're using a newer version of RDKit, LoadSDF might read the compounds that it wasn't reading when I ran this before
	for nsc in [119875, 266046, 753082]:
		new_row = {k: np.nan for k in sdf_df.columns.tolist()}
		new_row['NSC'] = str(nsc)
		if nsc == 119875:
			drug = '14913-33-8'
		else:
			drug = compound_names_df.loc[compound_names_df['nsc'] == nsc, 'name'].values[0]
		results = pcp.get_compounds(drug, 'name')

		if len(results) != 0:
			new_row['smiles'] = results[0].canonical_smiles
		sdf_df = sdf_df.append(new_row, ignore_index=True)
	sdf_df = sdf_df[['NSC', 'smiles']]

	# Preprocess SMILES strings:
	preprocessor = SmilesStandardizer()
	sdf_df = preprocessor.preprocess_df(sdf_df, 'smiles')

	# Add SMILES strings to the response data
	df = df.merge(sdf_df, how='left', left_on='NSC1', right_on='NSC')
	df.rename(columns={'smiles': 'SMILES_A'}, inplace=True)
	df = df.merge(sdf_df, how='left', left_on='NSC2', right_on='NSC')
	df.rename(columns={'smiles': 'SMILES_B'}, inplace=True)
	df.drop(['NSC_x', 'NSC_y'], axis=1, inplace=True)
	df.to_csv(output_path, index=False)


def preprocess_mut(pathway_level=False, pathway_counts=False, output_path=None):
	"""
	Preprocess mutation data for the NCI-60 cell lines used in the ALMANAC study. Removes silent mutations,
	binarizes mutations per gene and summarizes the data at the pathway-level if the pathway_level option is set."

	Parameters
	----------
	pathway_level: bool
		Summarize mutations at the pathway level instead of gene level.
	pathway_counts: bool
		Use counts to summarize mutations at the pathway level, counting each time a mutation occurs in a gene
		belonging to a given pathway, instead of using binary values to indicate the presence/absence of mutated genes
		belonging to a given pathway.
	output_path: str
		Path to the file where the preprocessed mutation data will be saved.

	Returns
	-------
	preprocessed_df: DataFrame
		The preprocessed mutation data.
	"""
	cols_to_use = ['Hugo_Symbol', 'Consequence', 'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode',
	               'CLIN_SIG', 'VARIANT_CLASS', 'GENE_PHENO', 'MA:FImpact', 'SIFT', 'GMAF', 'IMPACT', 'PolyPhen',
	               'FILTER']
	mut_df = pd.read_table('../data/cbioportal_nci60/data_mutations_extended.txt', header=1,
	                       usecols=cols_to_use)
	mut_df.rename(columns={'Tumor_Sample_Barcode': 'CELLNAME'}, inplace=True)

	# Map CBioportal cell line names to CellminerCDB cell line names:
	df_cellmap = pd.read_csv('../data/cbioportal_to_cellminercdb_cells.csv')
	df_cellmap.set_index('Cbioportal', inplace=True)
	cellmap = df_cellmap[['CellminerCDB']].to_dict()['CellminerCDB']
	mut_df['CELLNAME'] = mut_df['CELLNAME'].map(lambda x: cellmap[x])

	# Remove cell lines not used in ALMANAC:
	response_df = pd.read_csv(
		'../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		usecols=['CELLNAME'])
	almanac_cell_lines = response_df['CELLNAME'].unique().tolist()
	del response_df
	mut_df = mut_df[mut_df['CELLNAME'].isin(almanac_cell_lines)]

	# Remove silent mutations
	mut_df = mut_df[mut_df['Variant_Classification'].isin(
		['Splice_Region', 'Missense_Mutation', 'In_Frame_Del', 'Nonsense_Mutation', 'Frame_Shift_Del',
		 'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Ins', 'Translation_Start_Site', 'Nonstop_Mutation'])]

	# # Remove variants identified as common variants and variants with a frequency greater than or equal to 1% in 1000 Genomes
	# mut_df['GMAF'] = mut_df['GMAF'].str.split(':', expand=True)[1].astype(float)
	# mut_df = mut_df[~(mut_df['GMAF'] >= 0.01) & (mut_df['FILTER'] != 'common_variant')]
	#
	# # Filter based on the clinical significance of variant from dbSNP
	# mut_df = mut_df[~mut_df['CLIN_SIG'].isin(['benign', 'benign,likely_benign', 'likely_benign'])]
	#
	# # Filter according to SIFT, PolyPhen and VEP Impact predictions
	# mut_df = mut_df[
	# 	~(mut_df['SIFT'].str.contains('tolerated\(.+\)', regex=True, na=False)) & ~mut_df['PolyPhen'].str.contains(
	# 		'benign', na=False) & ~mut_df['IMPACT'].isin(['LOW', 'MODIFIER'])]

	mut_df['value'] = 1
	binarized_df = mut_df.reset_index().groupby(['CELLNAME', 'Hugo_Symbol'])['value'].aggregate(
		'first').unstack().fillna(0)
	binarized_df.columns.name = None

	if pathway_level:
		with open('../data/msigdb_c2.cp.reactome.v7.2.symbols.gmt', 'r') as f:
			lines = f.readlines()
		pathway_data_dict = defaultdict(list)
		for line in lines:
			line = line.strip().split('\t')
			pathway = line[0]
			pathway_genes = line[2:]
			if pathway_counts:
				for id, row in binarized_df.iterrows():
					mutated_genes = (row[row == 1].index.tolist())
					count = len(set(mutated_genes).intersection(set(pathway_genes)))
					pathway_data_dict[pathway].append(count)
			else:
				for id, row in binarized_df.iterrows():
					mutated_genes = (row[row == 1].index.tolist())
					if len(set(mutated_genes).intersection(set(pathway_genes))) > 0:
						pathway_data_dict[pathway].append(1)
					else:
						pathway_data_dict[pathway].append(0)
		preprocessed_df = pd.DataFrame(pathway_data_dict, index=binarized_df.index)
	else:
		preprocessed_df = binarized_df

	# remove columns that are constant (all zeroes or all ones):
	nonconstant_cols = preprocessed_df.columns[preprocessed_df.nunique() > 1]
	preprocessed_df = preprocessed_df[nonconstant_cols]
	preprocessed_df.reset_index(inplace=True)
	print(binarized_df.shape)
	print(preprocessed_df.shape)

	if output_path is not None:
		preprocessed_df.to_csv(output_path, index=False)

	return preprocessed_df


def preprocess_cnvs_gistic(prot_coding_only=False, output_path=None):
	"""
	Preprocess copy number variation (CNV) data (GISTIC scores) for the NCI-60 cell lines used in the ALMANAC study.

	Parameters
	----------
	prot_coding_only: bool
		Whether to only keep protein coding genes.
	output_path: str
		Path to the file where the preprocessed CNV data will be saved.

	Returns
	-------
	None

	"""
	cnv_df = pd.read_table('../data/cbioportal_nci60/data_CNA.txt', header=0).drop('Entrez_Gene_Id', axis=1).set_index(
		'Hugo_Symbol').T
	cnv_df.columns.name = None

	# Map CBioportal cell line names to CellminerCDB cell line names:
	df_cellmap = pd.read_csv('../data/cbioportal_to_cellminercdb_cells.csv')
	df_cellmap.set_index('Cbioportal', inplace=True)
	cellmap = df_cellmap[['CellminerCDB']].to_dict()['CellminerCDB']
	cnv_df.index = cnv_df.index.map(lambda x: cellmap[x])
	# Remove cell lines not used in ALMANAC:
	response_df = pd.read_csv(
		'../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		usecols=['CELLNAME'])
	almanac_cell_lines = response_df['CELLNAME'].unique().tolist()
	del response_df
	cnv_df = cnv_df[cnv_df.index.isin(almanac_cell_lines)]

	if prot_coding_only:
		prot_coding_df = pd.read_table('../data/hgnc_symbol_prot_coding.txt', header=0)
		prot_coding_list = prot_coding_df['Approved symbol'].dropna(axis=0).tolist() + prot_coding_df[
			'Alias symbol'].dropna(axis=0).tolist() + prot_coding_df['Previous symbol'].dropna(axis=0).tolist()
		genes_to_keep = list(set(cnv_df.columns.tolist()).intersection(set(prot_coding_list)))
		cnv_df = cnv_df[genes_to_keep]

	cnv_df.index.name = 'CELLNAME'
	cnv_df.reset_index(inplace=True)

	if output_path is not None:
		cnv_df.to_csv(output_path, index=False)


def preprocess_expr_rnaseq(expr_dataset_path='../data/CellminerCDB/data_nci60_xsq.txt', convert_to_tpm=False,
                           prot_coding_only=False, output_path=None):
	"""
	Preprocess RNA-Seq data (log2(FPKM+1)) for the NCI-60 cell lines used in the ALMANAC study.

	Parameters
	----------
	expr_dataset_path: str
		Path to the expression dataset file.
	convert_to_tpm: bool
		Whether to convert the log2(FPKM+1) values to log2(TPM+1).
	prot_coding_only: bool
		Whether to use protein coding genes only.
	output_path: str
		Path to the file where the preprocessed expression data will be saved.

	Returns
	-------
	expr_df: DataFrame
		The preprocessed expression data.

	"""
	expr_df = pd.read_table(expr_dataset_path, header=0)
	# RNA-seq composite log2 FPKM+1

	response_df = pd.read_csv(
		'../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		usecols=['CELLNAME'])
	cell_lines = response_df['CELLNAME'].unique().tolist()
	del response_df

	# expr_df = expr_df[expr_df['CELLNAME'].isin(cell_lines)].drop(['CELLNAME'], axis=1)
	# print(expr_df.columns[expr_df.nunique() <= 1])

	# Transpose df
	expr_df.set_index('Unnamed: 0', inplace=True)
	expr_df.rename(columns=lambda x: x.split(':')[-1], inplace=True)
	expr_df.index.name = None
	expr_df = expr_df[cell_lines]
	expr_df = expr_df.T

	if prot_coding_only:  # keep only protein-coding genes
		prot_coding_df = pd.read_table('../data/hgnc_symbol_prot_coding.txt', header=0)
		prot_coding_list = prot_coding_df['Approved symbol'].dropna(axis=0).tolist() + prot_coding_df[
			'Alias symbol'].dropna(axis=0).tolist() + prot_coding_df['Previous symbol'].dropna(axis=0).tolist()
		genes_to_keep = list(set(expr_df.columns.tolist()).intersection(set(prot_coding_list)))
		expr_df = expr_df[genes_to_keep]  # shape = [60 rows x 18788 columns]

	if convert_to_tpm:  # convert log2(FPKM+1) to log2(TPM+1)
		# based on: https://github.com/hosseinshn/MOLI/blob/master/preprocessing_scr/RNAseq.ipynb
		fpkm = expr_df.apply(np.exp2) - 1  # undo log transform and subtract 1 to get the original FPKM values
		sum_fpkm = fpkm.apply(sum, axis=0)
		tpm = fpkm / sum_fpkm * 1000000 + 1
		expr_df = tpm.applymap(np.log2)
		del fpkm, sum_fpkm, tpm

	expr_df.index.name = 'CELLNAME'
	expr_df.reset_index(inplace=True)

	# Remove constant columns
	constant_genes = expr_df.columns[expr_df.nunique() <= 1].tolist()
	print(constant_genes)
	print(expr_df.shape)
	expr_df.drop(constant_genes, axis=1, inplace=True)
	print(expr_df.shape)

	if output_path is not None:
		expr_df.to_csv(output_path, index=False)

	return expr_df


def get_drug_gene_interactions():
	"""Get the drug-gene interactions for the compounds in ALMANAC."""
	train_response_df = pd.read_csv(
		'../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_train.csv.gz')
	compound_names_df = pd.read_table('../data/ALMANAC/ComboCompoundNames_small.txt', header=None,
	                                  names=['nsc', 'name'])
	# print(compound_names_df[compound_names_df.duplicated('nsc', keep=False)]) # has duplicate entries, only going to keep the first ones
	compound_names_df.drop_duplicates(subset=['nsc'], keep='first', inplace=True)
	train_response_compounds = set(train_response_df['NSC1'].tolist() + train_response_df['NSC2'].tolist())
	compound_names_df = compound_names_df[compound_names_df['nsc'].isin(train_response_compounds)]
	compound_names_df['name'] = compound_names_df['name'].str.upper()
	compound_map = {'QUINACRINE HYDROCHLORIDE': 'QUINACRINE',
	                'PROCARBAZINE HYDROCHLORIDE': 'PROCARBAZINE',
	                'BLEOMYCIN SULFATE': 'BLEOMYCIN',
	                'BENDAMUSTINE HYDROCHLORIDE': 'BENDAMUSTINE',
	                'PEMETREXED DISODIUM ': 'PEMETREXED',
	                '2-FLUORO ARA-A': 'FLUDARABINE',
	                "4'-EPIADRIAMYCIN": 'EPIRUBICIN',
	                'SUNITINIB (FREE BASE)': 'SUNITINIB'}
	compound_names_df['name'].replace(compound_map, inplace=True)
	interactions_df = pd.read_table('../interactions.tsv') # file from DGIdb
	almanac_interactions_df = compound_names_df.merge(interactions_df, left_on='name', right_on='drug_name',
	                                                  how='left')  # creates multiple entries for each nsc
	# remove rows without gene_name
	print(almanac_interactions_df.shape)  # (3914, 13)
	almanac_interactions_df.dropna(subset=['gene_name'], inplace=True)
	print(almanac_interactions_df.shape)  # (3801, 13)
	almanac_interactions_df.to_csv('../data/almanac_drug_gene_interactions_full.csv', index=False)
	target_genes = almanac_interactions_df['gene_name'].unique().tolist()
	print(len(target_genes))  # 994 genes
	with open('../data/target_genes_full_list.txt', 'w') as f:
		for gene in target_genes:
			f.write(gene + '\n')
	almanac_interactions_df_reduced = almanac_interactions_df.dropna(subset=['interaction_types'])
	target_genes_reduced = almanac_interactions_df_reduced['gene_name'].unique().tolist()
	print(len(target_genes_reduced))
	with open('../data/target_genes_reduced_list.txt', 'w') as f:
		for gene in target_genes_reduced:
			f.write(gene + '\n')


def one_hot_encode_labels(response_dataset_path, output_dir):
	"""One-hot encodes cell line and drug names"""
	df = pd.read_csv(response_dataset_path)
	onehot_cells = pd.get_dummies(df['CELLNAME'])
	onehot_drugA = pd.get_dummies(df['NSC1'], prefix='NSC')
	onehot_drugB = pd.get_dummies(df['NSC2'], prefix='NSC')
	for drug in set(onehot_drugA.columns.tolist()).difference(set(onehot_drugB.columns.tolist())):
		# add NSCs in NSC1 column and not in NSC2 column
		onehot_drugB[drug] = 0
	for drug in set(onehot_drugB.columns.tolist()).difference(set(onehot_drugA.columns.tolist())):
		onehot_drugA[drug] = 0
	onehot_drugA = onehot_drugA.sort_index(axis=1)
	onehot_drugB = onehot_drugB.sort_index(axis=1)
	assert onehot_drugA.columns.tolist() == onehot_drugB.columns.tolist()
	onehot_cells.to_csv(os.path.join(output_dir, 'onehot_cell_line_names.csv'), index=False)
	onehot_drugA.to_csv(os.path.join(output_dir, 'onehot_drugA_NSCs.csv'), index=False)
	onehot_drugB.to_csv(os.path.join(output_dir, 'onehot_drugB_NSCs.csv'), index=False)
	# files need to be split before using


def extract_smiles_for_mtembeddings():
	df = pd.read_csv(
		'../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
	smiles = list(set(df['SMILES_A'].unique().tolist() + df['SMILES_B'].unique().tolist()))
	with open('../data/almanac_smiles.txt', 'w') as f:
		for i, compound in enumerate(smiles):
			if i != len(smiles) - 1:
				f.write(compound + '\n')
			else:
				f.write(compound)


if __name__ == '__main__':
	# Preprocess response dataset:
	preprocess_almanac_cellminercdb(output_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
	                                remove_duplicate_triplets=True)
	# Mutations, gene-level:
	preprocess_mut(pathway_level=False, pathway_counts=False, output_path='../data/nci_almanac_preprocessed/omics/unmerged/mut_gene_level_binarized.csv')
	# Mutations, pathway-level:
	preprocess_mut(pathway_level=True, pathway_counts=False, output_path='../data/nci_almanac_preprocessed/omics/unmerged/mut_pathway_level_binarized.csv')
	# CNVs:
	preprocess_cnvs_gistic(prot_coding_only=True, output_path='../data/nci_almanac_preprocessed/omics/unmerged/cnvs_gistic_prot_coding.csv')
	# Expression data (all genes):
	preprocess_expr_rnaseq(prot_coding_only=False, output_path='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_all.csv')
	# Expression data (protein coding only):
	preprocess_expr_rnaseq(prot_coding_only=True, output_path='../data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_prot_coding.csv')
	# Split drug response dataset into train/validation/test datasets and save:
	response_data_preprocessor = DatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
	response_data_preprocessor.split(
		split_inds_file='../data/splits/train_val_test_groups_split_inds_12321.pkl')
	response_data_preprocessor.save_split_datasets(output_dir='../data/nci_almanac_preprocessed/response',
	                                               output_name='almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples',
	                                               output_format='.csv.gz')
	# One-hot encoded IDs
	one_hot_encode_labels(
		response_dataset_path='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
		output_dir='../data/nci_almanac_preprocessed/onehot')