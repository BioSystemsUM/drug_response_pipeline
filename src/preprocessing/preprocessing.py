import os
import pickle
import sys
from copy import copy
import pkg_resources

import numpy as np
import pandas as pd
import pubchempy as pcp
import umap
from natsort import index_natsorted
from pyDeepInsight import ImageTransformer, LogScaler
from rdkit import Chem
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

from src.preprocessing import featurizers
from src.preprocessing.standardization import SmilesStandardizer

sys.setrecursionlimit(1000000)


class Conv1DReshaper(TransformerMixin, BaseEstimator):
	"""Add a dimension to the dataset so that it can be used as input to a 1D CNN."""

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		if isinstance(X, pd.DataFrame):
			X = X.values
		return np.expand_dims(X, axis=2)


class Conv2DReshaper(TransformerMixin, BaseEstimator):
	"""Reshape dataset so that it can be used as input to a 2D CNN"""
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		if isinstance(X, pd.DataFrame):
			X = X.values
		if X.shape[1] < self.width * self.height:
			ncols_to_add = (self.width * self.height) - X.shape[1]
			zeros = np.zeros((X.shape[0], ncols_to_add))  # add zeroes if necessary
			X = np.concatenate([X, zeros], axis=1)

		X = np.apply_along_axis(self._create_2d_grid, axis=1, arr=X)  # reshape each row

		return X

	def _create_2d_grid(self, example):
		example = np.reshape(example, newshape=(self.width, self.height))
		return np.expand_dims(example, axis=2)


class DatasetPreprocessor(object):
	"""General class to preprocess (merge, split) datasets read from CSV files."""

	def __init__(self, dataset_filepath, id_col=None):
		"""
		Parameters
		----------
		dataset_filepath: str
			Path to the dataset file.
		id_col: str
			The ID column
		"""
		self.dataset = self.load_data(dataset_filepath)
		self.split_datasets = None  # will be a dict with keys indicating train/val/test
		self.id_col = id_col

	def load_data(self, dataset_filepath):
		try:
			dataset = pd.read_csv(dataset_filepath)  # only accepting CSVs because I need to work with column names
			return dataset
		except Exception as e:
			print(e)
			return None

	def merge(self, dataset_to_merge_filepath):
		"""
		Merge with a response dataset

		Parameters
		----------
		dataset_to_merge_filepath: str
			Path to the response dataset file to be merged.

		Returns
		-------
		None
		"""
		response_df = pd.read_csv(dataset_to_merge_filepath, engine='python')
		drop_cols = response_df.columns.tolist()
		drop_cols.remove(self.id_col)
		merged_df = response_df.merge(self.dataset, how='left', on=self.id_col)
		merged_df.drop(drop_cols, axis=1, inplace=True)
		assert merged_df[self.id_col].tolist() == response_df[self.id_col].tolist(), "id_col order should be identical to the order of IDs found in the dataset we're merging with"
		del response_df
		self.dataset = merged_df

	def split(self, split_inds_file):
		"""
		Split the dataset.

		Parameters
		----------
		split_inds_file: str
			Path to file containing the train/validation/test split indices.

		Returns
		-------
		None
		"""
		with open(split_inds_file, 'rb') as f:
			inds_lists = pickle.load(f)

		if len(inds_lists) == 3:
			dataset_names = ['train', 'val', 'test']
		elif len(inds_lists) == 2:
			dataset_names = ['train', 'test']

		self.split_datasets = {}
		for i, ind_list in enumerate(inds_lists):
			df = copy(self.dataset).iloc[ind_list, :]
			self.split_datasets[dataset_names[i]] = df

	def save_split_datasets(self, output_dir, output_name, output_format, drop_id=False):
		"""
		Saves split datasets.

		Parameters
		----------
		output_dir: str
			Path to the directory where the split dataset files will be saved.
		output_name: str
			The name of the file (a suffix indicating whether the file is the training, validation or test set will
			be added to this string)
		output_format: str
			The file format.
		drop_id: bool
			Whether to drop the ID column or not.

		Returns
		-------
		None
		"""
		if drop_id:
			drop_cols = [self.id_col]
		else:
			drop_cols = None

		for key, value in self.split_datasets.items():
			filepath = os.path.join(output_dir, output_name + '_%s%s' % (key, output_format))
			self._save_to_file(dataset=value, output_filepath=filepath, cols_to_drop=drop_cols)

	def save_full_dataset(self, output_filepath, drop_id=False):
		"""
		Saves the full dataset.

		Parameters
		----------
		output_filepath: str
			Path where the file will be saved.
		drop_id: bool
			Whether to drop the ID column or not

		Returns
		-------
		None
		"""
		if drop_id:
			drop_cols = [self.id_col]
		else:
			drop_cols = None
		self._save_to_file(dataset=self.dataset, output_filepath=output_filepath, cols_to_drop=drop_cols)

	def _save_to_file(self, dataset, output_filepath, cols_to_drop=None):
		"""Save to CSV, CSV.GZ or npy file."""
		split_path = os.path.splitext(output_filepath)
		if split_path[-1] == '.gz':
			format = os.path.splitext(split_path[0])[-1] + split_path[-1]
		else:
			format = split_path[-1]

		if cols_to_drop is None:
			cols_to_drop = []

		if isinstance(dataset, pd.DataFrame):
			dataset = dataset.drop(cols_to_drop, axis=1, errors='ignore')

		if format == '.npy':
			if isinstance(dataset, pd.DataFrame):
				dataset = dataset.values
			np.save(output_filepath, dataset)
		elif format == '.csv' or format == '.csv.gz':
			if not isinstance(dataset, pd.DataFrame):
				cols = ['col_%s' % i for i in range(dataset.shape[1])]
				dataset = pd.DataFrame(data=dataset, columns=cols)
			if format == '.csv.gz':
				dataset.to_csv(output_filepath, index=False, compression='gzip')
			else:
				dataset.to_csv(output_filepath, index=False)
		else:
			raise ValueError('Output file format %s is not supported' % format)


class DrugDatasetPreprocessor(DatasetPreprocessor):
	"""Preprocess drug datasets"""

	def __init__(self, dataset_filepath, id_col=None, smiles_col=None):
		"""
		Parameters
		----------
		dataset_filepath: str
			Path to the dataset file.
		id_col: str
			The ID column.
		smiles_col:
			The column containing the SMILES strings.
		"""
		super().__init__(dataset_filepath=dataset_filepath, id_col=id_col)
		self.smiles_col = smiles_col

	def _get_unique_ids(self):
		return list(set(self.dataset[self.id_col].tolist()))

	def _get_unique_smiles(self):
		return list(set(self.dataset[self.smiles_col].tolist()))

	def _get_max_number_of_atoms(self):
		smiles_list = self._get_unique_smiles()
		num_atoms = [Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles_list]
		return max(num_atoms)

	def get_smiles_strings(self, smiles_col_name, dropna=True):
		"""
		Retrieve canonical SMILES strings for the molecules using PubChemPy.

		Parameters
		----------
		smiles_col_name: str
			The name that will be given to the SMILES column that will be added to the dataset.
		dropna: bool
			Whether to drop rows that are missing SMILES strings

		Returns
		-------
		None
		"""
		self.smiles_col = smiles_col_name
		drugs = self._get_unique_ids()
		smiles_dict = {'DRUG': [], smiles_col_name: []}
		for drug in drugs:
			smiles_dict['DRUG'].append(drug)
			results = pcp.get_compounds(drug, 'name')  # can return multiple results...
			if len(results) == 0:
				print(drug + ': no results')
				smiles_dict[smiles_col_name].append(np.nan)
			else:
				smiles_dict[smiles_col_name].append(results[0].canonical_smiles)
		smiles_df = pd.DataFrame(smiles_dict)
		self.dataset = self.dataset.merge(smiles_df, how='inner', left_on=self.id_col, right_on='DRUG').drop(['DRUG'], axis=1)

		if dropna: # drops rows that are missing SMILES strings
			self.dataset.dropna(axis=0, how='any', subset=[smiles_col_name], inplace=True)

	def standardize_smiles(self):
		"""Standardize the molecules"""
		if self.smiles_col is None:
			raise ValueError('SMILES column is undefined. Please define the column containing the SMILES'
							 'strings or get SMILES strings by calling the .get_smiles_strings method')
		else:
			standardizer = SmilesStandardizer()
			if self.smiles_col is not None:
				self.dataset = standardizer.preprocess_df(self.dataset, self.smiles_col)

	def featurize(self, featurizer_name, featurizer_args, output_dir, output_prefix=None, featurize_full_dataset=False,
				  featurize_split_datasets=True):
		"""
		Featurize the molecules.

		Parameters
		----------
		featurizer_name: str
			The name of the featurization method.
		featurizer_args: dict
			Arguments to be passed to the featurizer class.
		output_dir: str
			Path to directory where the featurized drug dataset will be saved
		output_prefix: str
			Prefix to add to the output filename
		featurize_full_dataset: bool
			Whether to featurize the full dataset.
		featurize_split_datasets: bool
			Whether to featurize the split dataset (the train/validation/test splits).

		Returns
		-------

		"""
		if self.smiles_col is None:
			raise ValueError('SMILES column is undefined. Please define the column containing the SMILES'
							 'strings or get SMILES strings by calling the .get_smiles_strings method')
		else:
			if output_prefix is None:
				output_prefix = featurizer_name

			if featurize_full_dataset:
				if featurizer_name == 'GraphFeaturizer':
					self._featurize_graphs(self.dataset, featurizer_args, output_dir=output_dir,
										   output_prefix=output_prefix)
				else:
					self._featurize_smiles(self.dataset, featurizer_name, featurizer_args, output_dir, output_prefix)

			if featurize_split_datasets:
				for dataset_name, dataset in self.split_datasets.items():
					new_output_prefix = '{prefix}_{name}'.format(prefix=output_prefix, name=dataset_name)
					if featurizer_name == 'GraphFeaturizer':
						self._featurize_graphs(dataset, featurizer_args, output_dir=output_dir,
											   output_prefix=new_output_prefix)
					else:
						self._featurize_smiles(dataset, featurizer_name, featurizer_args, output_dir, new_output_prefix)

	def _featurize_smiles(self, dataset, featurizer_name, featurizer_args, output_dir, output_prefix):
		featurizer = getattr(featurizers, featurizer_name)(**featurizer_args)
		featurized_df = featurizer.featurize_df(dataset, self.smiles_col)

		output_filepath = os.path.join(output_dir, output_prefix + '.npy')
		np.save(output_filepath, featurized_df.values)

		return featurized_df

	def _featurize_graphs(self, dataset, featurizer_args, output_dir, output_prefix):
		# TODO: try to join this code with self._featurize_smiles code??
		if 'max_num_atoms' not in featurizer_args:
			featurizer_args['max_num_atoms'] = self._get_max_number_of_atoms()
		featurizer = featurizers.GraphFeaturizer(**featurizer_args)
		prefix, dataset_name = output_prefix.split('_')
		if self.smiles_col is not None:
			featurizer.featurize_df(dataset, self.smiles_col,
			                        output_path_node_features=os.path.join(output_dir,
			                                                               '{prefix}_nodes_{name}.npy'.format(prefix=output_prefix, name=dataset_name)),
			                        output_path_adjacency_matrices=os.path.join(output_dir,
																				'{prefix}_adjmatrix_{name}.npy'.format(prefix=output_prefix, name=dataset_name)))

	def save_smiles_to_file(self, output_filepath):
		"""Save the unique SMILES strings in the dataset to a txt file."""
		smiles_list = self._get_unique_smiles()
		with open(output_filepath, 'w') as f:
			for i, smiles in enumerate(smiles_list):
				if i != len(smiles_list) - 1:
					f.write(smiles + '\n')
				else:
					f.write(smiles)


class OmicsDatasetPreprocessor(DatasetPreprocessor):
	"""Preprocess omics datasets."""

	def __init__(self, dataset_filepath, id_col=None):
		"""
		Parameters
		----------
		dataset_filepath: str
			Path to the dataset file. The rows must correspond to cell lines/patients and the columns must be genes.
		id_col: str
			The ID column.
		"""
		super().__init__(dataset_filepath=dataset_filepath, id_col=id_col)

	def _build_preprocessing_pipeline(self, variance_threshold=None, scaler=None, reshape_conv1d=False,
									  reshape_conv2d=False, conv_2d_shape=138, umap_embeddings=False,
									  umap_n_neighbors=20, umap_n_components=50, umap_use_densmap=False,
									  deepinsight_images=False, deepinsight_method='tsne', deepinsight_grid_size=50):
		"""Build the preprocessing pipeline"""
		steps = []

		if variance_threshold is not None:
			steps.append(VarianceThreshold(threshold=variance_threshold))

		if scaler is not None:
			if scaler == 'LogScaler':
				scaler = LogScaler()
			else:
				scaler = getattr(preprocessing, scaler)
			steps.append(scaler())

		# steps.append(SimpleImputer(strategy='median')) # not necessary right now, but may be necessary

		if reshape_conv1d:
			steps.append(Conv1DReshaper())

		if reshape_conv2d:
			steps.append(Conv2DReshaper(conv_2d_shape, conv_2d_shape))

		if umap_embeddings:
			steps.append(umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components,
								   densmap=umap_use_densmap, metric='correlation', random_state=12321))

		if deepinsight_images:
			steps.append(ImageTransformer(feature_extractor=deepinsight_method, pixels=deepinsight_grid_size,
										  n_jobs=10))

		if steps == []:
			return None
		else:
			return make_pipeline(*steps)

	def preprocess_split_datasets(self, variance_threshold=None, scaler=None,
									 reshape_conv1d=False, reshape_conv2d=False, conv_2d_shape=138,
									 umap_embeddings=False, umap_n_neighbors=20, umap_n_components=50,
									 umap_use_densmap=False, deepinsight_images=False, deepinsight_method='tsne',
									 deepinsight_grid_size=50):
		"""
		Preprocess the split datasets.

		Parameters
		----------
		variance_threshold: float or None
			The threshold for VarianceThreshold. VarianceThreshold will only be used if this is not None.
		scaler: str or None
			The scaling method that will be applied. Scaling will only be performed if this is not None.
		reshape_conv1d: bool
			If True, reshapes the data into a format suitable for 1D CNNs.
		reshape_conv2d: bool
			If True, reshapes the data into a format suitable for 2D CNNs.
		conv_2d_shape:
			A tuple containing the width and height of the images created when using Conv2DReshaper.
		umap_embeddings: bool
			If True, transforms the data into UMAP embeddings
		umap_n_neighbors: int
			The size of local neighborhood (in terms of number of neighboring sample points) used for manifold
			approximation in UMAP.
		umap_n_components: int
			The dimension of the space to embed into.
		umap_use_densmap: bool
			If True, uses Densmap instead of the original UMAP algorithm.
		deepinsight_images: bool
			If True, transforms the tabular omics data into images using Deepinsight.
		deepinsight_method: str
			The feature extraction method for Deepinsight. Only used if deepinsight_images is True.
		deepinsight_grid_size: int
			The size of the images created using Deepinsight. Only used if deepinsight_images is True.

		Returns
		-------
		dict
			The preprocessed split datasets.
		"""
		if self.split_datasets is None:
			print('The dataset has not been split yet. Call the .split() method first')
		else:
			pipeline = self._build_preprocessing_pipeline(variance_threshold, scaler, reshape_conv1d, reshape_conv2d,
														  conv_2d_shape, umap_embeddings, umap_n_neighbors,
														  umap_n_components, umap_use_densmap, deepinsight_images,
														  deepinsight_method, deepinsight_grid_size)
			train_df = self.split_datasets['train']
			train_df.drop([self.id_col], axis=1, inplace=True)
			pipeline.fit(train_df)

			preprocessed_datasets = {}

			for key, val in self.split_datasets.items():
				df = val.drop([self.id_col], axis=1, errors='ignore')
				preprocessed_datasets[key] = pipeline.transform(df)
				# np.save(os.path.join(output_dir, output_name + '_%s.npy' % key), preprocessed_datasets[key].values)
			self.split_datasets = preprocessed_datasets
			del preprocessed_datasets
			return self.split_datasets

	def preprocess_full_dataset(self, variance_threshold=None, scaler=None,
								reshape_conv1d=False, reshape_conv2d=False, conv_2d_shape=138,
								umap_embeddings=False, umap_n_neighbors=20, umap_n_components=50,
								umap_use_densmap=False, deepinsight_images=False, deepinsight_method='tsne',
								deepinsight_grid_size=50):
		"""
		Preprocess the dataset.

		Parameters
		----------
		variance_threshold: float or None
			The threshold for VarianceThreshold. VarianceThreshold will only be used if this is not None.
		scaler: str or None
			The scaling method that will be applied. Scaling will only be performed if this is not None.
		reshape_conv1d: bool
			If True, reshapes the data into a format suitable for 1D CNNs.
		reshape_conv2d: bool
			If True, reshapes the data into a format suitable for 2D CNNs.
		conv_2d_shape:
			A tuple containing the width and height of the images created when using Conv2DReshaper.
		umap_embeddings: bool
			If True, transforms the data into UMAP embeddings
		umap_n_neighbors: int
			The size of local neighborhood (in terms of number of neighboring sample points) used for manifold
			approximation in UMAP.
		umap_n_components: int
			The dimension of the space to embed into.
		umap_use_densmap: bool
			If True, uses Densmap instead of the original UMAP algorithm.
		deepinsight_images: bool
			If True, transforms the tabular omics data into images using Deepinsight.
		deepinsight_method: str
			The feature extraction method for Deepinsight. Only used if deepinsight_images is True.
		deepinsight_grid_size: int
			The size of the images created using Deepinsight. Only used if deepinsight_images is True.

		Returns
		-------
		dict
			The preprocessed dataset.
		"""
		pipeline = self._build_preprocessing_pipeline(variance_threshold, scaler, reshape_conv1d, reshape_conv2d,
													  conv_2d_shape, umap_embeddings, umap_n_neighbors,
													  umap_n_components, umap_use_densmap, deepinsight_images,
													  deepinsight_method, deepinsight_grid_size)

		full_df = self.dataset.drop([self.id_col], axis=1)
		pipeline.fit(full_df)
		self.dataset = pipeline.transform(full_df)
		del full_df
		return self.dataset

	def reorder_genes(self, gene_order_filepath=None, sort_alphabetically=False):
		"""

		Parameters
		----------
		gene_order_filepath
		sort_alphabetically

		Returns
		-------

		"""

		if (gene_order_filepath is not None) and (os.path.exists(gene_order_filepath)):
			with open(gene_order_filepath, 'rb') as f:
				cols_order = pickle.load(f)
		elif sort_alphabetically:
			cols_order = sorted(self.dataset.columns.tolist())
		else:
			cols_order = self.dataset.columns.tolist()

		if self.id_col is not None and self.id_col not in cols_order:
			cols_order.insert(0, self.id_col)

		self.dataset = self.dataset[cols_order]

	def filter_genes(self, use_targets=False, use_landmark=False, use_cosmic=False, use_ncg=False, use_msigdb=False,
					 use_aliases=True):
		"""
		Select genes using predefined gene lists.

		Parameters
		----------
		use_targets: bool
			If True, uses genes involved in known drug-gene interactions in DGIdb # TODO: make this more general in the future, as this only applies to ALMANAC drugs at the moment
		use_landmark: bool
			If True, uses landmark genes.
		use_cosmic: bool
			If True, uses genes from the COSMIC Cancer Gene Census
		use_ncg: bool
			If True, uses genes from the Network of Cancer Genes (NCG6.0)
		use_msigdb: bool
			If True, uses genes from the MSigDB C6 (oncogenic signature gene sets) and C4 (computational gene sets)
			collections
		use_aliases: bool
			Whether to use aliases for certain gene names

		Returns
		-------
		None
		"""
		# TODO: make this more general
		selected = [self.id_col]

		if use_targets:
			path = 'filtering_files/target_genes_full_list.txt'
			filepath = pkg_resources.resource_filename(__name__, path)
			with open(filepath, 'r') as f:
				target_genes = [line.rstrip() for line in f]
			target_aliases = {'SEM1': 'SHFM1', 'C10ORF67': 'LINC01552', 'MAP3K20': 'ZAK', 'NSD2': 'WHSC1',
							  'CYCSP5': 'HCP5'}
			for gene in target_genes:
				if use_aliases and gene in target_aliases:
					selected.append(target_aliases[gene])
				else:
					selected.append(gene)

		if use_landmark:
			path = 'filtering_files/lincs_landmark_list.txt'
			filepath = pkg_resources.resource_filename(__name__, path)
			with open(filepath, 'r') as f:
				landmark_genes = [line.rstrip() for line in f]
			landmark_aliases = {'COQ8A': 'ADCK3', 'WASHC5': 'KIAA0196', 'PRUNE1': 'PRUNE', 'TOMM70': 'TOMM70A',
								'CARMIL1': 'LRRC16A',
								'MINDY1': 'FAM63A', 'WASHC4': 'KIAA1033', 'KIF1BP': 'KIAA1279'}
			for gene in landmark_genes:
				if use_aliases and gene in landmark_aliases:
					selected.append(landmark_aliases[gene])
				else:
					selected.append(gene)

		if use_ncg:
			path = 'filtering_files/NCG6_cancergenes.tsv'
			filepath = pkg_resources.resource_filename(__name__, path)
			ncg_df = pd.read_table(filepath, header=0)
			ncg_genes = ncg_df['symbol'].unique().tolist()
			ncg_aliases = {'PATJ': 'INADL', 'BABAM2': 'BRE', 'ARMH3': 'C10orf76', 'C2CD6': 'ALS2CR11',
						   'WDCP': 'C2orf44',
						   'SPART': 'SPG20', 'HIKESHI': 'C11orf73', 'CCNQ': 'FAM58A', 'BICRA': 'GLTSCR1',
						   'PCNX4': 'PCNXL4',
						   'TWNK': 'C10orf2', 'RFLNA': 'FAM101A', 'GSDME': 'DFNA5', 'NSD2': 'WHSC1L1',
						   'ATP5F1B': 'ATP5B',
						   'DCAF1': 'VPRBP', 'KNL1': 'CASC5', 'AFDN': 'MLLT4', 'NSD3': 'WHSC1L1', 'LHFPL6': 'LHFP',
						   'ELOC': 'TCEB1', 'THAP12': 'PRKRIR', 'CNOT9': 'RQCD1', 'FYB2': 'C1orf168', 'RTL9': 'RGAG1',
						   'MAP3K21': 'KIAA1804', 'PAK5': 'PAK7', 'ERBIN': 'ERBB2IP'}
			for gene in ncg_genes:
				if use_aliases and gene in ncg_aliases:
					selected.append(ncg_aliases[gene])
				else:
					selected.append(gene)

		if use_cosmic:
			path = 'filtering_files/cancer_gene_census.csv'
			filepath = pkg_resources.resource_filename(__name__, path)
			cosmic_df = pd.read_csv(filepath)
			cosmic_genes = cosmic_df['Gene Symbol'].unique().tolist()  # 723 genes
			cosmic_aliases = {'MRTFA': 'MKL1', 'TENT5C': 'FAM46C', 'NSD3': 'WHSC1L1', 'WDCP': 'C2orf44',
							  'LHFPL6': 'LHFP', 'AFDN': 'MLLT4', 'NSD2': 'WHSC1', 'KNL1': 'CASC5'}
			for gene in cosmic_genes:
				if use_aliases and gene in cosmic_aliases:
					selected.append(cosmic_aliases[gene])
				else:
					selected.append(gene)

		if use_msigdb:
			msigdb_genes = []
			path = 'filtering_files/msigdb_c6.all.v7.4.symbols.gmt'
			filepath = pkg_resources.resource_filename(__name__, path)
			with open(filepath, 'r') as f:
				lines = f.readlines()
			for line in lines:
				line = line.strip().split('\t')
				msigdb_genes.extend(line[2:])
			path = 'filtering_files/msigdb_c4.all.v7.4.symbols.gmt'
			filepath = pkg_resources.resource_filename(__name__, path)
			with open(filepath, 'r') as f:
				for line in lines:
					line = line.strip().split('\t')
					msigdb_genes.extend(line[2:])
			selected.extend(msigdb_genes)

		selected = list(set(selected))
		# Some genes in the dataset may be using previous gene symbols instead of the most current ones:
		path = 'filtering_files/hgnc_symbol_prot_coding.txt'
		filepath = pkg_resources.resource_filename(__name__, path)
		hgnc_df = pd.read_table(filepath, header=0)
		genes_names_not_in_df = list(set(selected).difference(set(self.dataset.columns.tolist())))
		hgnc_df = hgnc_df[hgnc_df.isin(genes_names_not_in_df).any(axis=1)]
		genes_previous_symbols = hgnc_df['Previous symbol'].dropna(axis=0).tolist()
		selected.extend(genes_previous_symbols)
		selected = list(set(selected))

		self.dataset = self.dataset.loc[:, self.dataset.columns.isin(selected)]

	def get_clustering_gene_order(self, output_filepath, method='complete', metric='correlation',
								  scaling_method='MinMaxScaler'):
		"""Use hierarchical clustering to find a new order for the genes"""
		# it's better if we do this with the unmerged file
		df = self.dataset.drop([self.id_col], axis=1)
		labels = df.columns.tolist()
		if scaling_method is not None:
			print('scaling')
			scaler = self._build_preprocessing_pipeline(scaler=scaling_method)
			df = scaler.fit_transform(df)
		else:
			df = df.values
		df = np.transpose(df)
		print('creating distance matrix')
		dist_mat = pdist(df, metric=metric)
		del df
		print('hierarchical clustering')
		hc = linkage(dist_mat, method=method, metric=metric, optimal_ordering=True)
		print('creating dendrogram')
		d = dendrogram(hc, labels=labels, no_plot=True)
		del hc
		with open(output_filepath, 'wb') as f:
			pickle.dump(d['ivl'], f)

	def get_chromosome_gene_order(self, output_filepath):
		"""Order of the genes according to their chromosome positions."""
		# TODO: improve code. right now it's very specific to the ALMANAC dataset and it's confusing
		# Replace HDGFRP3 with HDGFL3:
		omics_genes = ['HDGFL3' if gene == 'HDGFRP3' else gene for gene in self.dataset]
		path = 'filtering_files/ensembl_gene_chromosome_pos.csv'
		filepath = pkg_resources.resource_filename(__name__, path)
		ensembl_df = pd.read_csv(filepath,
		                         usecols=['Chromosome/scaffold name', 'Gene start (bp)', 'Gene end (bp)', 'Strand',
										  'HGNC symbol', 'Gene name', 'Gene Synonym'])
		path = 'filtering_files/hgnc_genes_chromosomes.txt'
		filepath = pkg_resources.resource_filename(__name__, path)
		hgnc_df = pd.read_table(filepath, header=0)
		gene_map = {'Gene': omics_genes, 'HGNC': []}
		for gene in omics_genes:
			match_df = hgnc_df[hgnc_df.eq(gene).any(1)]
			if not match_df.empty:
				hgnc_symbol = match_df.at[match_df.index[0], 'Approved symbol']
				gene_map['HGNC'].append(hgnc_symbol)
			else:
				gene_map['HGNC'].append(np.nan)
		genes_df = pd.DataFrame(data=gene_map)
		genes_df = genes_df.merge(hgnc_df.drop_duplicates(subset=['Approved symbol'], keep='first'), left_on='HGNC',
								  right_on='Approved symbol', how='left')

		df1 = genes_df.merge(ensembl_df.drop_duplicates(subset=['HGNC symbol'], keep='first'), left_on='HGNC',
							 right_on='HGNC symbol')
		remaining1 = list(set(omics_genes).difference(df1['Gene'].tolist()))
		df2 = genes_df[genes_df['Gene'].isin(remaining1)].merge(
			ensembl_df.drop_duplicates(subset=['Gene name'], keep='first'), left_on='Gene', right_on='Gene name')
		remaining2 = list(set(omics_genes).difference(set(df1['Gene'].tolist() + df2['Gene'].tolist())))
		df3 = genes_df[genes_df['Gene'].isin(remaining2)].merge(
			ensembl_df.drop_duplicates(subset=['Gene Synonym'], keep='first'), left_on='Gene', right_on='Gene Synonym')
		positions_df = pd.concat([df1, df2, df3], axis=0)
		missing = list(set(omics_genes).difference(set(positions_df['Gene'].tolist())))
		missing_df = genes_df[genes_df['Gene'].isin(missing)]
		positions_df = pd.concat([positions_df, missing_df], axis=0)
		positions_df['Chromosome'].fillna(positions_df['Chromosome/scaffold name'], inplace=True)
		positions_df.sort_values(['Chromosome', 'Gene start (bp)'], ascending=[True, True],
								 key=lambda x: np.argsort(index_natsorted(x)),
								 inplace=True)
		positions_df.to_csv('chromosome_order_prot_coding.csv', index=False)
		sorted_genes = positions_df['Gene'].tolist()
		with open(output_filepath, 'wb') as f:
			pickle.dump(sorted_genes, f)