import copy
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spektral.utils import pad_jagged_array
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.scoring import scoring_metrics



def save_evaluation_results(results_dict, hyperparams, model_descr, model_dir, output_filepath=None):
    """
    Save model evaluation results and best hyperparameter values to a CSV file.

    Parameters
    ----------
    results_dict: dict
        Dictionary containing the model performance scores.
    hyperparams: dict
        The hyperparameters and their respective values.
    model_descr: str
        Model description.
    model_dir:
        Directory where the trained model was saved.
    output_filepath: str
        Path to the CSV file.

    Returns
    -------
    None
    """
    results = {'model': model_descr, 'model_dir': model_dir}
    results.update({k: [v] for k, v in hyperparams.items()})
    results.update({k: [v] for k, v in results_dict.items()})
    results_df = pd.DataFrame(data=results)
    if output_filepath is None:
        evaluation_output = os.path.join(model_dir, 'evaluation_results.csv')
    else:
        evaluation_output = output_filepath

    if os.path.exists(evaluation_output):
        saved_df = pd.read_csv(evaluation_output)
        results_df = pd.concat([saved_df, results_df], axis=0, ignore_index=True, sort=False)

    # change column order
    cols = results_df.columns.tolist()
    score_cols = sorted(results_dict.keys())
    hyperparam_cols = sorted([x for x in cols if x not in ['model', 'model_dir'] + score_cols])
    rearranged_cols = ['model', 'model_dir'] + hyperparam_cols + score_cols
    results_df = results_df[rearranged_cols]
    results_df.to_csv(evaluation_output, index=False)


def plot_keras_history(history, output_dir):
    """Plots training history for Keras models."""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'model_loss.png'), format='png')


def build_char_dict(dataset_path, smiles_cols, save=True):
    """Builds the character dictionary required when using TextCNN models.

    Based on code from DeepChem (https://github.com/deepchem/deepchem/blob/master/deepchem/models/text_cnn.py)"""
    default_dict = {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9,
                    '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17,
                    'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25,
                    ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33}
    # SMILES strings
    smiles_df = pd.read_csv(dataset_path)
    smiles_list = []
    for col in smiles_cols:
        smiles_list.extend(smiles_df[col].values.tolist())
    smiles_list = list(set(smiles_list))  # only keeping unique SMILES
    # Maximum length is expanded to allow length variation during train and inference
    seq_length = int(max([len(smile) for smile in smiles_list]) * 1.2)
    # '_' served as delimiter and padding
    all_smiles = '_'.join(smiles_list)
    tot_len = len(all_smiles)
    # Initialize common characters as keys
    keys = list(default_dict.keys())
    out_dict = copy.deepcopy(default_dict)
    current_key_val = len(keys) + 1
    # Include space to avoid extra keys
    keys.extend([' '])
    extra_keys = []
    i = 0
    while i < tot_len:
        # For 'Cl', 'Br', etc.
        if all_smiles[i:i + 2] in keys:
            i = i + 2
        elif all_smiles[i:i + 1] in keys:
            i = i + 1
        else:
            # Character not recognized, add to extra_keys
            extra_keys.append(all_smiles[i])
            keys.append(all_smiles[i])
            i = i + 1
    # Add all extra_keys to char_dict
    for extra_key in extra_keys:
        out_dict[extra_key] = current_key_val
        current_key_val += 1

    if save:
        with open('../char_dict_seq_len.pkl', 'wb') as f:
            pickle.dump((out_dict, seq_length), f)

    return out_dict, seq_length


def get_max_number_of_atoms(smiles_list):
    """Returns the maximum number of atoms seen in a list of SMILES strings"""
    num_atoms = [MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles_list]
    return max(num_atoms)


def zero_pad_graphs(n_max, x_list=None, a_list=None, e_list=None):
    """
    Zero-pad molecular graphs.

    Copied this function from spektral. Modified the function so that the user can specify n_max

    Converts lists of node features, adjacency matrices and edge features to
    [batch mode](https://graphneural.network/data-modes/#batch-mode),
    by zero-padding all tensors to have the same node dimension `n_max`.
    Either the node features or the adjacency matrices must be provided as input.
    The i-th element of each list must be associated with the i-th graph.
    If `a_list` contains sparse matrices, they will be converted to dense
    np.arrays.
    The edge attributes of a graph can be represented as
    - a dense array of shape `(n_nodes, n_nodes, n_edge_features)`;
    - a sparse edge list of shape `(n_edges, n_edge_features)`;
    and they will always be returned as dense arrays.
    :param x_list: a list of np.arrays of shape `(n_nodes, n_node_features)`
    -- note that `n_nodes` can change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(n_nodes, n_nodes)`;
    :param e_list: a list of np.arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`;
    :return: only if the corresponding list is given as input:
        -  `x`: np.array of shape `(batch, n_max, n_node_features)`;
        -  `a`: np.array of shape `(batch, n_max, n_max)`;
        -  `e`: np.array of shape `(batch, n_max, n_max, n_edge_features)`;
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list")

    # n_max = max([x.shape[0] for x in (x_list if x_list is not None else a_list)])

    # Node features
    x_out = None
    if x_list is not None:
        x_out = pad_jagged_array(x_list, (n_max, -1))

    # Adjacency matrix
    a_out = None
    if a_list is not None:
        if hasattr(a_list[0], "toarray"):  # Convert sparse to dense
            a_list = [a.toarray() for a in a_list]
        a_out = pad_jagged_array(a_list, (n_max, n_max))

    # Edge attributes
    e_out = None
    if e_list is not None:
        if e_list[0].ndim == 2:  # Sparse to dense
            for i in range(len(a_list)):
                a, e = a_list[i], e_list[i]
                e_new = np.zeros(a.shape + e.shape[-1:])
                e_new[np.nonzero(a)] = e
                e_list[i] = e_new
        e_out = pad_jagged_array(e_list, (n_max, n_max, -1))

    return tuple(out for out in [x_out, a_out, e_out] if out is not None)


def get_dataset_dims(filepaths_dict, model_type):
    """Returns the dimensions of each dataset in 'filepaths_dict'"""
    dims = {}
    for key, filepath in filepaths_dict.items():
        if filepath is not None:
            if 'drug' in key:
                new_key = 'drug_dim'
            else:
                new_key = key + '_dim'
            try:
                arr = np.load(filepath)
                if (key == 'expr') and ('expr2dconv' in model_type):
                    dims[new_key] = arr[0].shape
                else:
                    dims[new_key] = arr[0].shape[0]
            except: # this will happen when file is not .npy
                dims[new_key] = None
    return dims


def get_ml_algorithm(model_type):
    """
    Gets a machine learning model by name

    Parameters
    ----------
    model_type: str
        The name of the model

    Returns
    -------
    estimator object

    """
    models_dict = {'RandomForestRegressor': RandomForestRegressor(),
                   'ElasticNet': ElasticNet(),
                   'SVR': SVR(),
                   'LinearSVR': LinearSVR(),
                   'XGBRegressor': XGBRegressor(),
                   'LGBMRegressor': LGBMRegressor(),
                   'Nystroem+LinearSVR': Pipeline([('nystroem', Nystroem()), ('linearsvr', LinearSVR())])}
    return models_dict[model_type]


def evaluate_ml_model(model, test_dataset, metrics):
    """
    Evaluates traditional machine learning models.

    Parameters
    ----------
    model: estimator object
        The trained model.
    test_dataset: MultiInputDataset object
        The multi-input test dataset.
    metrics: list
        List of scoring metrics.

    Returns
    -------
    scores_dict: dict
        Dictionary with the model performance results.

    """
    scores_dict = {}
    y_pred = model.predict(test_dataset.X)
    for metric_name in metrics:
        metric_func = getattr(scoring_metrics, metric_name)
        scores_dict[metric_name] = metric_func(test_dataset.y, y_pred)
    return scores_dict


def plot_morgan_bits(smiles_file, smiles_col, sample_id, bits, fp_radius, fp_length, drug_img_filepath,
                     bits_img_filepath, prefix=None):
    """
    Draw 2D depictions of user-defined Morgan (ECFP) fingerprint bits that are 'ON' (set to 1) in a given molecule.

    Parameters
    ----------
    smiles_file: str
        Path to file with molecules represented as SMILES strings.
    smiles_col: str
        Column containing the SMILES strings.
    sample_id: int
        The row containing the molecule for which the bits will be plotted.
    bits: list
        The Morgan fingerprint bits that will be plotted.
    fp_radius:
        The Morgan fingerprint radius.
    fp_length:
        The length of the fingerprint (number of bits).
    drug_img_filepath:
        Path to save the image of the full molecule.
    bits_img_filepath:
        Path to save the 2D depictions of the user-selected 'ON' bits.
    prefix: str
        String to add to the bit labels.

    Returns
    -------
    None

    """
    test_response_data = pd.read_csv(smiles_file)
    sample = test_response_data.loc[sample_id, :]
    drug = Chem.MolFromSmiles(sample[smiles_col])
    Draw.MolToFile(drug, drug_img_filepath, size=(500, 500))
    drug_bi = {}
    drug_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(drug, radius=fp_radius, nBits=fp_length, bitInfo=drug_bi)
    draw_opt = MolDrawOptions()
    draw_opt.padding = 0.1
    draw_opt.legendFontSize = 24
    draw_opt.minFontSize = 12
    draw_opt.bondLineWidth = 3
    draw_opt.centreMoleculesBeforeDrawing = True
    if prefix == None:
        prefix='bit_'
    drug_img = Draw.DrawMorganBits([(drug, x, drug_bi) for x in bits],
                                    molsPerRow=5,
                                    legends=[prefix+str(x + 1) for x in bits],
                                    useSVG=True,
                                    subImgSize=(350, 350),
                                    drawOptions=draw_opt)
    with open(bits_img_filepath, 'w') as f:
        f.write(drug_img)