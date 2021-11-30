import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold

from src.dataset.dataset import MultiInputDataset


class TrainTestSplit(object):
	"""Random train-test split"""
	def __init__(self, test_size=0.1, random_state=None, split_inds_file=None):
		self.test_size = test_size
		self.stratify = None
		self.random_state = random_state
		if split_inds_file is not None:
			with open(split_inds_file, 'rb') as f:
				self.split_inds = pickle.load(f)
		else:
			self.split_inds = None


	def split(self, multi_input_dataset, mode):

		if self.split_inds is not None:
			train_inds, test_inds = self.split_inds
			X_train = multi_input_dataset.response_dataset.iloc[train_inds, :]
			y_train = multi_input_dataset.y.iloc[train_inds, :]
			X_test = multi_input_dataset.response_dataset.iloc[test_inds, :]
			y_test = multi_input_dataset.y.iloc[test_inds, :]
		else:
			if mode == 'classification':
				self.stratify = multi_input_dataset.y

			X_train, X_test, y_train, y_test = train_test_split(multi_input_dataset.response_dataset,
			                                                    multi_input_dataset.y,
			                                                    test_size=self.test_size,
			                                                    stratify=self.stratify,
			                                                    random_state=self.random_state)
			train_inds = y_train.index.tolist()
			test_inds = y_test.index.tolist()
			self.split_inds = (train_inds, test_inds)

		# split dataset_dict
		X_train_dict = {}
		X_test_dict = {}
		for key in multi_input_dataset.X_dict.keys():
			if type(multi_input_dataset.X_dict[key]) == np.ndarray:
				X_train_dict[key] = multi_input_dataset.X_dict[key][train_inds]
				X_test_dict[key] = multi_input_dataset.X_dict[key][test_inds]
			else:
				X_train_dict[key] = multi_input_dataset.X_dict[key].loc[train_inds, :]
				X_test_dict[key] = multi_input_dataset.X_dict[key].loc[test_inds, :]
		X_train_dataset = MultiInputDataset().create_from_dict(X_train_dict, X_train, y_train)
		X_test_dataset = MultiInputDataset().create_from_dict(X_test_dict, X_test, y_test)

		return [X_train_dataset, X_test_dataset]

	def save_split_inds(self, pickle_path):
		with open(pickle_path, 'wb') as f:
			pickle.dump(self.split_inds, f)


class TrainValTestSplit(object):
	"""Random train-validation-test split"""

	def __init__(self, val_size=0.1, test_size=0.1, random_state=None, split_inds_file=None):
		self.val_size = val_size
		self.test_size = test_size
		self.stratify = None
		self.random_state = random_state

		if split_inds_file is not None:
			with open(split_inds_file, 'rb') as f:
				self.split_inds = pickle.load(f)
		else:
			self.split_inds = None

	def split(self, multi_input_dataset, mode):

		if self.split_inds is not None:
			train_inds, val_inds, test_inds = self.split_inds
			X_train = multi_input_dataset.response_dataset.iloc[train_inds, :]
			y_train = multi_input_dataset.y[train_inds]
			X_val = multi_input_dataset.response_dataset.iloc[val_inds, :]
			y_val = multi_input_dataset.y[val_inds]
			X_test = multi_input_dataset.response_dataset.iloc[test_inds, :]
			y_test = multi_input_dataset.y[test_inds]
		else:
			if mode == 'classification':
				self.stratify = multi_input_dataset.y

			X_train, X_val_test, y_train, y_val_test = train_test_split(multi_input_dataset.response_dataset,
			                                                            multi_input_dataset.y,
			                                                            test_size=self.test_size,
			                                                            stratify=self.stratify,
			                                                            random_state=self.random_state)
			X_val, X_test, y_val, y_test = train_test_split(X_val_test,
			                                                y_val_test,
			                                                test_size=self.test_size / (self.test_size + self.val_size),
			                                                random_state=self.random_state)
			train_inds = y_train.index.tolist()
			val_inds = y_val.index.tolist()
			test_inds = y_test.index.tolist()
			self.split_inds = (train_inds, val_inds, test_inds)

		# split dataset_dict
		X_train_dict = {}
		X_val_dict = {}
		X_test_dict = {}
		for key in multi_input_dataset.X_dict.keys():
			if type(multi_input_dataset.X_dict[key]) == np.ndarray:
				X_train_dict[key] = multi_input_dataset.X_dict[key][train_inds]
				X_val_dict[key] = multi_input_dataset.X_dict[key][val_inds]
				X_test_dict[key] = multi_input_dataset.X_dict[key][test_inds]
			else:
				X_train_dict[key] = multi_input_dataset.X_dict[key].loc[train_inds, :]
				X_val_dict[key] = multi_input_dataset.X_dict[key].loc[val_inds, :]
				X_test_dict[key] = multi_input_dataset.X_dict[key].loc[test_inds, :]
		X_train_dataset, X_val_dataset, X_test_dataset = (MultiInputDataset(), MultiInputDataset(), MultiInputDataset())
		X_train_dataset.create_from_dict(X_train_dict, X_train, y_train)
		X_val_dataset.create_from_dict(X_val_dict, X_val, y_val)
		X_test_dataset.create_from_dict(X_test_dict, X_test, y_test)

		return [X_train_dataset, X_val_dataset, X_test_dataset]

	def save_split_inds(self, pickle_path):
		with open(pickle_path, 'wb') as f:
			pickle.dump(self.split_inds, f)


class KFoldSplit(object):
	def __init__(self, n_splits=5, random_state=None, split_inds_file=None):
		if split_inds_file is not None:
			with open(split_inds_file, 'rb') as f:
				self.split_inds = pickle.load(f)
			self.n_splits = len(self.split_inds)
		else:
			self.split_inds = None
			self.n_splits = n_splits
		self.random_state = random_state

	def split(self, multi_input_dataset, mode):
		if self.split_inds is None:
			if mode == 'classification':
				splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
			else:
				splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
		self.split_inds = list(splitter.split(multi_input_dataset.response_dataset, multi_input_dataset.y))

		# create split datasets
		kfold_datasets = []
		for i, (train_inds, val_inds) in enumerate(self.split_inds):
			X_train = multi_input_dataset.response_dataset.iloc[train_inds, :]
			y_train = multi_input_dataset.y.iloc[train_inds, :]
			X_val = multi_input_dataset.response_dataset.iloc[val_inds, :]
			y_val = multi_input_dataset.y.iloc [val_inds, :]

			X_train_dict = {}
			X_val_dict = {}
			for key in multi_input_dataset.X_dict.keys():
				if type(multi_input_dataset.X_dict[key]) == np.ndarray:
					X_train_dict[key] = multi_input_dataset.X_dict[key][train_inds]
					X_val_dict[key] = multi_input_dataset.X_dict[key][train_inds]
				else:
					X_train_dict[key] = multi_input_dataset.X_dict[key].loc[train_inds, :]
					X_val_dict[key] = multi_input_dataset.X_dict[key].loc[train_inds, :]

			X_train_dataset = MultiInputDataset().create_from_dict(X_train_dict, X_train, y_train)
			X_val_dataset = MultiInputDataset().create_from_dict(X_val_dict, X_val, y_val)

			kfold_datasets.append((X_train_dataset, X_val_dataset))

		return kfold_datasets

	def save_split_inds(self, pickle_path):
		with open(pickle_path, 'wb') as f:
			pickle.dump(self.split_inds, f)


def create_LDCO_splits_cellminercdb(random_seed, output_path):
	# for train-val-test
	# 80%/10%/10% split, where each set has different drug combinations not included in the other sets
	df = pd.read_csv('../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
	gkf = GroupKFold(n_splits=10)
	np.random.seed(seed=random_seed)
	groups = df['GROUP'].tolist()
	np.random.shuffle(groups)
	split_inds_folds = list(gkf.split(X=df, groups=groups))
	# 1st test fold will be test set
	# 2nd test fold will be validation set
	# 3rd, 4th and 5th test folds will be train set
	# doesn't work, because it's not guaranteed that every group will be a part of the test set at least once
	# "Each subject is in a different testing fold, and the same subject is never in both testing and training. Notice that the folds do not have exactly the same size due to the imbalance in the data"
	test_inds = split_inds_folds[0][1].tolist()
	val_inds = split_inds_folds[1][1].tolist()
	train_inds = []
	for i in range(2, 10):
		train_inds.extend(split_inds_folds[i][1].tolist())

	# print('confirm')
	# print(len(test_inds)) # 30001
	# print(len(val_inds)) # 30007
	# print(len(train_inds)) # 239943
	# print(df.shape) # (299951, 9)
	# print(len(test_inds)+len(val_inds)+len(train_inds)) # 299951
	# print(len(list(set(test_inds).intersection((set(val_inds)))))) # 0
	# print(len(list(set(test_inds).intersection((set(train_inds)))))) # 0
	# print(len(list(set(val_inds).intersection((set(train_inds)))))) # 0
	# print(sorted(train_inds+val_inds+test_inds) == sorted(df.index.tolist())) # True

	with open(output_path, 'wb') as f:
		pickle.dump([train_inds, val_inds, test_inds], f)

if __name__ == '__main__':
	create_LDCO_splits_cellminercdb(random_seed=12321, output_path='../data/splits/train_val_test_groups_split_inds_12321.pkl')
