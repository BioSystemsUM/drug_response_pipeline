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
        """
        Featurize a DataFrame of molecules represented as SMILES strings.

        Parameters
        ----------
        smiles_df: DataFrame
            DataFrame with the SMILES strings of the molecules that will be featurized
        smiles_col: str
            The name of the column containing the SMILES strings
        output_path: str
            Path to the CSV file where the featurized molecules will be saved.

        Returns
        -------
        featurized_df: DataFrame
            A pandas DataFrame containing the featurized molecules.
        """
        featurized_df = smiles_df[smiles_col].apply(self.featurize_molecule).apply(convert_numpy_to_list).apply(
            pd.Series)
        featurized_df = featurized_df.rename(columns=lambda x: self.featurizer_name + '_' + str(x + 1))

        if output_path is not None:
            featurized_df.to_csv(output_path, index=False)

        return featurized_df

    def featurize_molecule(self, smiles_string):
        """
        Featurize a single molecule.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule

        Raises
        -------
        NotImplementedError:
            This method is not implemented here since each subclass will have a different implementation.
        """
        raise NotImplementedError('Featurization method is not defined.')


class ECFPFeaturizer(BaseFeaturizer):
    """Compute ECFP (Morgan) fingerprints for molecules."""

    def __init__(self, radius, length=2048):
        """
        Parameters
        ----------
        radius: int
            The radius of the atom environments. For ECFP4 fingerprints, use radius=2. For ECFP6, use a radius of 3.
        length:
            Number of bits in the fingerprint.
        """
        self.radius = radius
        self.length = length
        if self.radius == 2:
            self.featurizer_name = 'ECFP4'
        elif self.radius == 3:
            self.featurizer_name = 'ECFP6'

    def featurize_molecule(self, smiles_string):
        """
        Compute ECFP fingerprint for a single molecule.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule

        Returns
        -------
        arr: array
            The ECFP fingerprint for the molecule.
        """
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
    """Compute RDKit layered fingerprints for molecules."""

    def __init__(self, layer_flags=4294967295, min_path=1, max_path=7, fp_size=2048, atom_counts=[],
                 branched_paths=True):
        """
        Parameters
        ----------
        layer_flags: int
             The layers to include in the fingerprint.
        min_path: int
            Minimum number of bonds to include in the subgraphs
        max_path: int
            Maximum number of bonds to include in the subgraphs
        fp_size: int
            Number of bits in the fingerprint.
        atom_counts: list
            A list of the number of paths that set bits each atom is involved in. Should be at least as long as the
            number of atoms in the molecule.
        branched_paths: bool
            If True, both branched and unbranched paths will be used in the fingerprint.
        """
        self.layer_flags = layer_flags
        self.min_path = min_path
        self.max_path = max_path
        self.fp_size = fp_size
        self.atom_counts = atom_counts
        self.branched_paths = branched_paths
        self.featurizer_name = 'LayeredFP'

    def featurize_molecule(self, smiles_string):
        """
        Compute the layered fingerprint for a single molecule.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        arr: array
            The layered fingerprint for the molecule.
        """
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
    """Compute Mol2vec embeddings for molecules using a pretrained model."""

    def __init__(self, trained_model='../mol2vec_models/model_300dim.pkl'):
        """
        Parameters
        ----------
        trained_model: str
            Path to a pretrained Mol2vec model.
        """
        self.featurizer_name = 'mol2vec'
        self.model = Word2Vec.load(trained_model)

    def featurize_molecule(self, smiles_string):
        """

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        array
            Mol2vec embedding of the molecule.

        """
        mol_obj = Chem.MolFromSmiles(smiles_string)
        sentence = mol2alt_sentence(mol_obj, 1)
        sentence2vec = sentences2vec([sentence], self.model, unseen='UNK')
        vector = DfVec(sentence2vec).vec
        return vector[0]


class TextCNNFeaturizer(BaseFeaturizer):
    """Encode SMILES strings so that they can fed into TextCNN models."""

    def __init__(self, char_dict, seq_length):
        """

        Parameters
        ----------
        char_dict: dict
            Dictionary mapping characters in SMILES strings to integers
        seq_length: int
            Maximum sequence length
        """
        self.featurizer_name = 'tokenized'
        self.char_dict = char_dict
        self.seq_len = seq_length

    def featurize_molecule(self, smiles_string):
        """
        Tokenize characters in a single SMILES string.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        array
            The tokenized SMILES.
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
    """Convert molecules represented as SMILES strings into molecular graphs. Each graph is defined using an adjacency
    matrix and the atom representations.

    Based on code from DeepChem: https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/mol_graph_conv_featurizer.py"""

    def __init__(self, zero_pad=True, max_num_atoms=96, normalize_adj_matrix=False, use_graphconv_featurization=False,
                 use_chirality=False, use_partial_charge=False):
        """
        Parameters
        ----------
        zero_pad: bool
            If True, zero-pads the graphs.
        max_num_atoms: int
            The maximum number of atoms allowed.
        normalize_adj_matrix: bool
            If True, the normalization described in the GCN (Kipf et al, 2017) paper will be applied.
        use_graphconv_featurization: bool
            If True, calculates the same atom features as DeepChem's ConvMolFeaturizer. Otherwise, it will calculate the
            atom features as defined for DeepChem's MolGraphConvFeaturizer.
        use_chirality: bool
            Whether to use chirality information or not.
        use_partial_charge: bool
            Whether to use partial charge data or not.
        """
        self.featurizer_name = 'graph'
        self.zero_pad = zero_pad
        self.max_num_atoms = max_num_atoms
        self.use_graphconv_featurization = use_graphconv_featurization
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality
        self.normalize_adj = normalize_adj_matrix

    def featurize_df(self, smiles_df, smiles_col, output_path_node_features=None, output_path_adjacency_matrices=None):
        """
        Convert a DataFrame containing molecules represented as SMILES strings into the graph representation required by
        graph neural networks.

        Parameters
        ----------
        smiles_df: DataFrame
            DataFrame with the SMILES strings of the molecules that will be featurized
        smiles_col: str
            The name of the column containing the SMILES strings
        output_path_node_features
            Path to the file in which the atom features for all of the molecules will be saved.
        output_path_adjacency_matrices:
            Path to the file in which the adjacency matrices for all of the molecules will be saved.

        Returns
        -------
        node_features: array
            The atom features for all of the molecules in the DataFrame.
        adjacency_matrices: array
            The adjacency matrices for all of the molecules in the DataFrame.
        """
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
        """
        Featurizes a single molecule for use with graph neural networks.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        node_features: array
            The atom features calculated for the molecule.
        adj_matrix: array
            The adjacency matrix for the molecule.
        """
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
    """Obtain Molecular Transformer Embeddings (MTEs) for molecules.

    The embeddings must first be generated using the embed.py script provided here:
    https://github.com/mpcrlab/MolecularTransformerEmbeddings

    The final embedding is a flattened version of the embedding obtained using the MTE model (flattened by taking the
    mean across the first dimension of the embedding).
    """

    def __init__(self, embeddings_file='../data/nci_almanac_preprocessed/drugs/mtembeddings_almanac_smiles.npz'):
        """
        Parameters
        ----------
        embeddings_file: str
            Path to file containing the already embedded SMILES strings. The file maps each SMILES string to its
            Molecular Transformer Embedding.
        """
        self.embeddings = np.load(embeddings_file)
        self.featurizer_name = 'MTEmbeddings'

    def featurize_molecule(self, smiles_string):
        """
        Embed a single molecule.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        embedding_flattened: array
            A flattened version of the Molecular Transformer Embedding of the molecule.
        """
        embedding = self.embeddings[smiles_string]
        # embedding_flattened = np.reshape(embedding, (-1, embedding.shape[0]*embedding.shape[1]))
        embedding_flattened = embedding.mean(axis=0)
        return embedding_flattened
