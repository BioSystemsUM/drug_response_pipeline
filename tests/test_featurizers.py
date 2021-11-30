from scipy.stats import mode
import pandas as pd

smiles = 'CC(=O)OC1(C(C)=O)CCC2C3C=C(C)C4=CC(=O)CCC4(C)C3CCC21C'
smiles2 = 'O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1'
smiles_df = pd.DataFrame(data={'smiles': ['CCO', 'CC(=O)C']})
# smiles_df = pd.read_csv('test_graph_conv_response_file.csv')
# smiles_df = smiles_df.sample(frac=0.1)
# smiles_df.to_csv('gcn_test_response_data.csv', index=False)
# print(smiles_df.shape)
# num_atoms = []
# for compound in set(smiles_df['SMILES_A'].tolist() + smiles_df['SMILES_B'].tolist()):
#     mol = Chem.MolFromSmiles(compound)
#     num_atoms.append(mol.GetNumAtoms())
#     adj_mat = Chem.GetAdjacencyMatrix(mol)
#     adj_mat2 = gcn_filter(adj_mat)
# # print(min(num_atoms))
# # print(max(num_atoms))
# # print(np.mean(num_atoms))
# # print(mode(num_atoms))
# gcn1 = GraphFeaturizer(max_num_atoms=max(num_atoms), use_graphconv_featurization=False)
# # n, adj = gcn1.featurize_molecule(smiles)
# #print(n.shape)
# gcn1.featurize_df(smiles_df, 'SMILES_A', output_path_node_features='smiles_a_nodes_test.npy',
#                   output_path_adjacency_matrices='smiles_a_adj_mat_test.npy')
# gcn1.featurize_df(smiles_df, 'SMILES_B', output_path_node_features='smiles_b_nodes_test.npy',
#                   output_path_adjacency_matrices='smiles_b_adj_mat_test.npy')
#
# gcn2 = GraphFeaturizer(max_num_atoms=max(num_atoms), use_graphconv_featurization=True)
# n, adj = gcn2.featurize_molecule(smiles)
# print(n.shape)