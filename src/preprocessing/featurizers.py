import numpy as np
import pandas as pd
from deepchem.feat.molecule_featurizers.mol_graph_conv_featurizer import _construct_atom_feature, \
    construct_hydrogen_bonding_info
from deepchem.feat.graph_features import atom_features
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, DfVec, sentences2vec
from rdkit import Chem, DataStructs
from rdkit.Chem import LayeredFingerprint
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from spektral.utils import gcn_filter

from src.utils.utils import zero_pad_graphs


def convert_numpy_to_list(np_array):
    return np_array.tolist()


class BaseFeaturizer(object):
    """Base class for chemical compound featurizers"""

    def __init__(self):
        self.featurizer_name = None

    def featurize_df(self, smiles_df, smiles_col, output_path=None):
        featurized_df = smiles_df[smiles_col].apply(self.featurize_molecule).apply(convert_numpy_to_list).apply(
            pd.Series)
        featurized_df = featurized_df.rename(columns=lambda x: self.featurizer_name + '_' + str(x + 1))

        if output_path is not None:
            featurized_df.to_csv(output_path, index=False)

        return featurized_df

    def featurize_molecule(self, smiles_string):
        raise NotImplementedError('Featurization method is not defined.')


class ECFPFeaturizer(BaseFeaturizer):

    def __init__(self, radius, length=2048):
        self.radius = radius
        self.length = length
        if self.radius == 2:
            self.featurizer_name = 'ECFP4'
        elif self.radius == 3:
            self.featurizer_name = 'ECFP6'

    def featurize_molecule(self, smiles_string):
        try:
            arr = np.zeros((1,))
            fp = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_string),
                                               self.radius, nBits=self.length)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except Exception as e:
            print(e)
            arr = np.nan
        return arr


class LayeredFPFeaturizer(BaseFeaturizer):

    def __init__(self, layer_flags=4294967295, min_path=1, max_path=7, fp_size=2048, atom_counts=[],
                 branched_paths=True):
        self.layer_flags = layer_flags
        self.min_path = min_path
        self.max_path = max_path
        self.fp_size = fp_size
        self.atom_counts = atom_counts
        self.branched_paths = branched_paths
        self.featurizer_name = 'LayeredFP'

    def featurize_molecule(self, smiles_string):
        try:
            arr = np.zeros((1,))
            fp = LayeredFingerprint(Chem.MolFromSmiles(smiles_string), layerFlags=self.layer_flags,
                                    minPath=self.min_path, maxPath=self.max_path, fpSize=self.fp_size,
                                    atomCounts=self.atom_counts, branchedPaths=self.branched_paths)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except Exception as e:
            print(e)
            print('error ' + smiles_string)
            arr = np.nan
        return arr


class Mol2VecFeaturizer(BaseFeaturizer):

    def __init__(self, trained_model='../mol2vec_models/model_300dim.pkl'):
        self.featurizer_name = 'mol2vec'
        self.model = Word2Vec.load(trained_model)

    def featurize_molecule(self, smiles_string):
        mol_obj = Chem.MolFromSmiles(smiles_string)
        sentence = mol2alt_sentence(mol_obj, 1)
        sentence2vec = sentences2vec([sentence], self.model, unseen='UNK')
        vector = DfVec(sentence2vec).vec
        return vector[0]


class TextCNNFeaturizer(BaseFeaturizer):

    def __init__(self, char_dict, seq_length):
        self.featurizer_name = 'tokenized'
        self.char_dict = char_dict
        self.seq_len = seq_length

    def featurize_molecule(self, smiles_string):
        """ Tokenize characters in smiles_string to integers
        """
        smiles_len = len(smiles_string)
        seq = [0]
        keys = self.char_dict.keys()
        i = 0
        while i < smiles_len:
            # Skip all spaces
            if smiles_string[i:i + 1] == ' ':
                i = i + 1
            # For 'Cl', 'Br', etc.
            elif smiles_string[i:i + 2] in keys:
                seq.append(self.char_dict[smiles_string[i:i + 2]])
                i = i + 2
            elif smiles_string[i:i + 1] in keys:
                seq.append(self.char_dict[smiles_string[i:i + 1]])
                i = i + 1
            else:
                raise ValueError('character not found in dict')
        for i in range(self.seq_len - len(seq)):
            # Padding with '_'
            seq.append(self.char_dict['_'])
        return np.array(seq, dtype=np.int32)


class GraphFeaturizer(BaseFeaturizer):

    def __init__(self, zero_pad=True, max_num_atoms=96, normalize_adj_matrix=False, use_graphconv_featurization=False,
                 use_chirality=False, use_partial_charge=False):
        self.featurizer_name = 'graph'
        self.zero_pad = zero_pad
        self.max_num_atoms = max_num_atoms
        self.use_graphconv_featurization = use_graphconv_featurization
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality
        self.normalize_adj = normalize_adj_matrix

    def featurize_df(self, smiles_df, smiles_col, output_path_node_features=None, output_path_adjacency_matrices=None):
        node_features = []
        adjacency_matrices = []
        for smiles in smiles_df[smiles_col].tolist():
            nodes, adjacency_matrix = self.featurize_molecule(smiles)
            node_features.append(nodes)
            adjacency_matrices.append(adjacency_matrix)

        if self.zero_pad:
            # zero-padding here because otherwise I can't save as numpy arrays (this is what Spektral's BatchLoader does)
            node_features, adjacency_matrices = zero_pad_graphs(self.max_num_atoms, node_features,
                                                                              adjacency_matrices)

        if output_path_node_features is not None:
            np.save(output_path_node_features, node_features)

        if output_path_adjacency_matrices is not None:
            np.save(output_path_adjacency_matrices, adjacency_matrices)

        return node_features, adjacency_matrices

    def featurize_molecule(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)

        # Atom featurization is based on DeepChem code

        if self.use_graphconv_featurization:
            new_order = Chem.CanonicalRankAtoms(mol)
            mol = Chem.RenumberAtoms(mol, new_order)
            idx_nodes = [(a.GetIdx(), atom_features(a, use_chirality=False)) for a in mol.GetAtoms()]
            idx_nodes.sort()
            idx, nodes = list(zip(*idx_nodes))
            node_features = np.vstack(nodes)
        else:
            if self.use_partial_charge:
                try:
                    mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
                except:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(mol)
            # construct atom (node) features
            h_bond_infos = construct_hydrogen_bonding_info(mol)
            node_features = np.asarray(
                [_construct_atom_feature(atom, h_bond_infos, self.use_chirality, self.use_partial_charge) for atom in
                 mol.GetAtoms()], dtype=float)

        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        if self.normalize_adj:
            adj_matrix = gcn_filter(adj_matrix)  # applies normalization described in Kipf et al, 2017 paper

        return node_features, adj_matrix


class MTEmbeddingsFeaturizer(BaseFeaturizer):

    def __init__(self, embeddings_file='../data/nci_almanac_preprocessed/drugs/mtembeddings_almanac_smiles.npz'):
        self.embeddings = np.load(embeddings_file)
        self.featurizer_name = 'MTEmbeddings'

    def featurize_molecule(self, smiles_string):
        embedding = self.embeddings[smiles_string]
        # embedding_flattened = np.reshape(embedding, (-1, embedding.shape[0]*embedding.shape[1]))
        embedding_flattened = embedding.mean(axis=0)
        return embedding_flattened
