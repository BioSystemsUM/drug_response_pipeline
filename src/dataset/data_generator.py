import os
import gc
import numpy as np
import pandas as pd
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generate data for Keras.

    Based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, y_filepath, output_col, drugA_filepath, drugB_filepath, drugA_atom_feat_filepath, drugB_atom_feat_filepath,
                 drugA_adj_filepath, drugB_adj_filepath, expr_filepath, mut_filepath, cnv_filepath, batch_size, shuffle=True):
        """
        Parameters
        ----------
        y_filepath: str
            Path to the file containing the output variable.
        output_col: str
            The name of the output variable.
        drugA_filepath: str
            Path to the file with drug features for drugA in each combination.
        drugB_filepath: str
            Path to the file with drug features for drugB in each combination.
        drugA_atom_feat_filepath: str
            Path to the file containing the atom (node) features for drugA in each combination. Only used for
            graph neural networks.
        drugB_atom_feat_filepath: str
            Path to the file containing the atom (node) features for drugB in each combination. Only used for
            graph neural networks.
        drugA_adj_filepath: str
            Path to file containing the adjacency matrices for drugA in each combination. Only used for graph
            neural networks.
        drugB_adj_filepath: str
            Path to file containing the adjacency matrices for drugB in each combination. Only used for graph
            neural networks.
        expr_filepath: str
            Path to the gene expression file.
        mut_filepath: str
            Path to the mutations file.
        cnv_filepath: str
            Path to the CNVs file.
        batch_size: int
            Batch size.
        shuffle: bool
            If true, shuffles the batch indices after each epoch.
        """
        self.output = pd.read_csv(y_filepath)[output_col].values
        self.n_rows = self.output.shape[0]
        self.indices = list(range(self.n_rows))
        self.X_filepaths = {'drugA': drugA_filepath,
                            'drugB': drugB_filepath,
                            'drugA_atom_feat': drugA_atom_feat_filepath,
                            'drugB_atom_feat': drugB_atom_feat_filepath,
                            'drugA_adj': drugA_adj_filepath,
                            'drugB_adj': drugB_adj_filepath,
                            'expr': expr_filepath,
                            'mut': mut_filepath,
                            'cnv': cnv_filepath}
        self.memmaps = {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_rows / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # index is the batch index (from 0 to # of batches per epoch (defined by __len__))
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        """Generates data containing batch_size samples"""
        X = {}
        y = self.output[batch_indices]

        # Generate data
        for key, filepath in self.X_filepaths.items():
            if filepath is not None:
                # if os.path.split(os.getcwd())[-1] != 'src':
                #     filepath = os.path.join('../..', '..', '..', '..', filepath)

                if key not in self.memmaps:
                    file = np.load(filepath, mmap_mode='r')
                    self.memmaps[key] = file
                else:
                    file = self.memmaps[key]
                X[key] = file[batch_indices]
        return X, y

    def delete_memmaps(self):
        del self.memmaps
        gc.collect()
