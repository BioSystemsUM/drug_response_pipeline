import os
import pickle

import numpy as np
import pandas as pd


class MultiInputDataset(object):
	"""
	Class to load and store multi-input datasets.
	"""

	def __init__(self, response_dataset_path=None, id_cols=None, output_col=None, input_order=None):
		""""""
		if response_dataset_path is not None:
			self.response_dataset, self.y = self.load_response_file(response_dataset_path, output_col)
		else:
			self.response_dataset = None
			self.y = None
		if id_cols is not None:
			self.id_cols = id_cols
		elif self.response_dataset is not None:
			self.id_cols = self.response_dataset.columns.tolist()
		else:
			self.id_cols = []

		self.X_dict = {}
		self.X_list = None
		self.X = None  # self.X will be a 2D numpy array with all features concatenated
		self.feature_names = None # will be a dict
		self.input_order = input_order

	def load_response_file(self, response_dataset_path, output_col):
		"""Loads response file and extracts the output variable"""
		response_df = pd.read_csv(response_dataset_path)
		y = response_df[output_col].values
		return response_df, y

	def load_drugA(self, drugA_file):
		"""Load featurized data for drug A"""
		if drugA_file is not None:
			self.X_dict['drugA'] = self._load_features(drugA_file, dtype=None)

	def load_drugB(self, drugB_file):
		"""Load featurized data for drug B"""
		if drugB_file is not None:
			self.X_dict['drugB'] = self._load_features(drugB_file, dtype=None)

	def load_graph_data(self, nodes_file, adj_file):
		"""Load nodes and adjacency matrix files """
		if (nodes_file is not None) and (adj_file is not None):
			drug_letter = os.path.split(os.path.splitext(nodes_file)[0])[-1].split('_')[1][-1]
			self.X_dict['drug%s_atom_feat' % drug_letter] = self._load_features(nodes_file)
			self.X_dict['drug%s_adj' % drug_letter] = self._load_features(adj_file)

	def load_expr(self, rnaseq_file):
		"""Load RNA-seq gene expression file"""
		if rnaseq_file is not None:
			self.X_dict['expr'] = self._load_features(rnaseq_file, dtype=np.float32)

	def load_mut(self, mut_file):
		"""Load mutation file"""
		if mut_file is not None:
			self.X_dict['mut'] = self._load_features(mut_file, dtype=np.int8)

	def load_cnv(self, cnv_file):
		"""Load copy number variation file"""
		if cnv_file is not None:
			self.X_dict['cnv'] = self._load_features(cnv_file, dtype=np.int8)  # np.int8 assuming this is the GISTIC file

	def _load_features(self, filepath, dtype=None):
		"""Load features from .csv or .npy files"""
		if '.csv' in filepath:
			if dtype is not None:
				data = pd.read_csv(filepath, dtype=dtype, memory_map=True)
			else:
				data = pd.read_csv(filepath, memory_map=True)
			data.drop(self.id_cols, axis=1, inplace=True, errors='ignore')
		elif '.npy' in filepath:
			data = np.load(filepath)
		else:
			raise ValueError('File format is not supported.')
		return data

	def load_dataset_dict(self, dataset_dict_path):
		"""Loads the dataset dictionary (X_dict) from a pickle file."""
		with open(dataset_dict_path, 'rb') as f:
			self.X_dict = pickle.load(f)
		return self.X_dict

	def load_y(self, y_path):
		"""Loads the output variable from a pickle file."""
		with open(y_path, 'rb') as f:
			self.y = pickle.load(f)
		return self.y

	def load_feature_names(self, feature_names_filepath):
		with open(feature_names_filepath, 'rb') as f:
			self.feature_names = pickle.load(f)
		return self.feature_names # will be a dict

	def feature_names_to_list(self):
		feature_names_list = []
		for input_type in self.input_order:
			feature_names_list.extend(self.feature_names[input_type])
		return feature_names_list


	def X_dict_to_list(self):
		"""Converts X_dict into a list maintaining the specified input (dict keys) order"""
		self.X_list = []
		if self.input_order is None:
			self.input_order = self.X_dict.keys()
		for key in self.input_order:
			self.X_list.append(self.X_dict[key])
		return self.X_list

	def create_from_dict(self, dataset_dict, y, response_df=None):
		"""Creates a MultiInputDataset from a dictionary"""
		self.X_dict = dataset_dict
		self.y = y
		self.response_dataset = response_df

	def select(self, indices):
		selected_X_dict = {}
		for key, val in self.X_dict.items():
			selected_X_dict[key] = val[indices]
		selected_y = self.y[indices]
		selected_response_dataset = self.response_dataset.iloc[indices, :]
		new_dataset = MultiInputDataset(id_cols=self.id_cols, input_order=self.input_order)
		new_dataset.create_from_dict(selected_X_dict, selected_y, selected_response_dataset)
		return new_dataset

	def sample(self, n=100):
		if n > self.response_dataset.shape[0]:
			raise ValueError('n must be smaller than {0}'.format(self.response_dataset.shape[0]))
		selected_inds = np.random.choice(self.response_dataset.shape[0], n, replace=False)
		new_dataset = self.select(selected_inds)
		return new_dataset

	def concat_features(self):
		"""Concatenate all of the input datasets"""
		if self.input_order is not None:
			keys = self.input_order
		else:
			keys = sorted(list(self.X_dict.keys()))  # this guarantees that the order the features appear in the final dataset is always the same, even if the datasets are loaded into X_dict in a different order between different runs
		for i, key in enumerate(keys):
			if i == 0:
				arr = self.X_dict[key]
			else:
				arr = np.concatenate((arr, self.X_dict[key]), axis=1)
		self.X = arr
		return self.X

	def concat_datasets(self, dataset_to_concat):
		""""Concatenate two multi-input datasets"""
		# dataset to concat must have the same keys as self.X_dict
		new_multiinputdataset = MultiInputDataset()
		new_X_dict = {}
		for i, (key, val) in enumerate(self.X_dict.items()):  # join X_dicts
			new_X_dict[key] = np.concatenate((val, dataset_to_concat.X_dict[key]), axis=0)
		new_y = np.concatenate((self.y, dataset_to_concat.y), axis=0)
		new_response_df = pd.concat([self.response_dataset, dataset_to_concat.response_dataset], axis=0)
		new_multiinputdataset.create_from_dict(dataset_dict=new_X_dict, y=new_y, response_df=new_response_df)
		return new_multiinputdataset

	def get_dataset_dimensions(self, model_type):
		"""Gets the shape of each of the individual datasets"""
		dims = {}
		for key, dataset in self.X_dict.items():
			if dataset is not None:
				if 'drug' in key:
					new_key = 'drug_dim'
				else:
					new_key = key + '_dim'
				if isinstance(dataset, pd.DataFrame):
					dataset = dataset.values
				if (key == 'expr') and ('expr2dconv' in model_type):
					dims[new_key] = dataset[0].shape
				else:
					dims[new_key] = dataset[0].shape[0]
		return dims

	def get_n_rows(self):
		"""Gets the number of rows"""
		return self.response_dataset.shape[0]

	def get_row_ids(self, sep='_'):
		row_ids_df = self.response_dataset[self.id_cols].astype(str)
		row_ids_df['concat_ids'] = row_ids_df.agg(sep.join, axis=1)
		return row_ids_df['concat_ids'].values

	def save_dataset_dict(self, output_filepath):
		"""Saves the multi-input dataset dictionary (X_dict) as a pickle file."""
		with open(output_filepath, 'wb') as f:
			pickle.dump(self.X_dict, f)

	def save_y(self, output_filepath):
		"""Saves the output variable as a pickle file."""
		with open(output_filepath, 'wb') as f:
			pickle.dump(self.y, f)
