import pickle
import numpy as np
import pandas as pd
from src.preprocessing.preprocessing import DrugDatasetPreprocessor, OmicsDatasetPreprocessor, DatasetPreprocessor


def test_dataset_preprocessor():
    dataset_preprocessor = DatasetPreprocessor('test_files/response_data_small.csv')

    dataset_preprocessor.split('test_files/test_split_inds.pkl')
    print(dataset_preprocessor.split_datasets)
    dataset_preprocessor.save_split_datasets(output_dir='test_files', output_name='split_data',
                                             output_format='.csv.gz')
    df = pd.read_csv('test_files/split_data_test.csv.gz')
    print(df)
    print(df.shape)

    dataset_preprocessor.save_full_dataset(output_filepath='test_files/full_data.npy')
    arr = np.load('test_files/full_data.npy', allow_pickle=True)
    print(arr)
    print(arr.shape)


def test_drug_preprocessor():
    print('drug A')
    drug_preprocessor = DrugDatasetPreprocessor(dataset_filepath='test_files/response_data_small.csv',
                                                id_col='NSC1',
                                                smiles_col='SMILES_A')
    max_n_atoms_A = drug_preprocessor._get_max_number_of_atoms()
    print(max_n_atoms_A)
    print(drug_preprocessor._get_unique_smiles())
    print(drug_preprocessor._get_unique_ids())
    drug_preprocessor.standardize_smiles()
    drug_preprocessor.save_smiles_to_file('test_files/smiles_drugA.txt')
    drug_preprocessor.split(split_inds_file='test_files/test_split_inds.pkl')
    print(drug_preprocessor.split_datasets)

    drug_preprocessor.featurize('ECFPFeaturizer', featurizer_args={'radius': 2, 'length': 1024},
                                output_dir='test_files',
                                output_prefix='ECFP4_drugA', featurize_split_datasets=True, featurize_full_dataset=False)

    print('drug B')
    drug_preprocessor = DrugDatasetPreprocessor(dataset_filepath='test_files/response_data_small.csv',
                                                id_col='NSC2',
                                                smiles_col='SMILES_B')
    max_n_atoms_B = drug_preprocessor._get_max_number_of_atoms()
    print(max_n_atoms_B)
    print(drug_preprocessor._get_unique_smiles())
    print(drug_preprocessor._get_unique_ids())
    drug_preprocessor.standardize_smiles()
    drug_preprocessor.save_smiles_to_file('test_files/smiles_drugB.txt')
    drug_preprocessor.split(split_inds_file='test_files/test_split_inds.pkl')
    print(drug_preprocessor.split_datasets)
    drug_preprocessor.featurize('ECFPFeaturizer', featurizer_args={'radius': 2, 'length': 1024},
                                output_dir='test_files',
                                output_prefix='ECFP4_drugB', featurize_split_datasets=True, featurize_full_dataset=False)

    # if we want to featurize for graph neural networks and we have 2 drug columns, we need to use the max number of atoms in the whole dataset
    max_n_atoms_overall = max(max_n_atoms_A, max_n_atoms_B)
    drug_preprocessor.featurize('GraphFeaturizer',
                                featurizer_args={'max_num_atoms': max_n_atoms_overall, 'normalize_adj_matrix':True},
                                output_dir='test_files',
                                output_prefix='GCN_drugB',
                                featurize_split_datasets=True,
                                featurize_full_dataset=False)
    # TODO: need to test .get_smiles_strings


def test_omics_preprocessor():
    omics_preprocessor = OmicsDatasetPreprocessor(dataset_filepath='../almanac/data/nci_almanac_preprocessed/omics/unmerged/rnaseq_fpkm_prot_coding.csv',
                                                  id_col='CELLNAME')
    print(omics_preprocessor.dataset)

    omics_preprocessor.filter_genes(use_targets=False, use_landmark=False, use_cosmic=True, use_ncg=False,
                                    use_msigdb=False)
    omics_preprocessor.save_full_dataset(output_filepath='test_files/expr_filtered_cosmic_unmerged.csv')
    df = pd.read_csv('test_files/expr_filtered_cosmic_unmerged.csv')
    print(df.shape)
    print(df.columns)

    omics_preprocessor.get_chromosome_gene_order(output_filepath='test_files/chromosome_order.pkl')
    with open('test_files/chromosome_order.pkl', 'rb') as f:
        genes = pickle.load(f)
    print(genes)

    omics_preprocessor.get_clustering_gene_order('test_files/clustering_order.pkl')
    with open('test_files/clustering_order.pkl', 'rb') as f:
        genes = pickle.load(f)
    print(genes)

    print(omics_preprocessor.dataset.columns)
    omics_preprocessor.reorder_genes(gene_order_filepath='test_files/clustering_order.pkl')
    print(omics_preprocessor.dataset.columns)

    print(omics_preprocessor.dataset)
    omics_preprocessor.merge(dataset_to_merge_filepath='test_files/response_data_small.csv')
    print(omics_preprocessor.dataset)
    response_df = pd.read_csv('test_files/response_data_small.csv')
    print(response_df['CELLNAME'].tolist() == omics_preprocessor.dataset['CELLNAME'].tolist())

    omics_preprocessor.save_full_dataset('test_files/expr_filtered_merged_no_id_col.csv', drop_id=True)
    omics_preprocessor.save_full_dataset('test_files/expr_filtered_merged.csv')

    omics_preprocessor.split(split_inds_file='test_files/test_split_inds.pkl')
    print(omics_preprocessor.split_datasets)
    omics_preprocessor.save_split_datasets(output_dir='test_files', output_name='expr', output_format='.csv.gz')
    omics_df = pd.read_csv('test_files/expr_train.csv.gz')
    print(omics_df)
    print(omics_df.columns.tolist())
    response_df = pd.read_csv('test_files/split_data_train.csv.gz')
    print(omics_df['CELLNAME'].tolist() == response_df['CELLNAME'].tolist())

    omics_preprocessor.preprocess_split_datasets(scaler='MinMaxScaler')
    omics_preprocessor.save_split_datasets(output_dir='test_files', output_name='expr_scaled', output_format='.csv.gz')
    omics_scaled_train_df_csv = pd.read_csv('test_files/expr_scaled_train.csv.gz')
    print(omics_scaled_train_df_csv)
    omics_preprocessor.save_split_datasets(output_dir='test_files', output_name='expr_scaled', output_format='.npy',
                                           drop_id=True)
    omics_scaled_train_npy = np.load('test_files/expr_scaled_train.npy')
    print(omics_scaled_train_npy)

    scaled_dataset = omics_preprocessor.preprocess_full_dataset(scaler='MinMaxScaler')
    print(scaled_dataset)
    omics_preprocessor.save_full_dataset('test_files/expr_scaled_full_dataset.npy', drop_id=True)
    omics_scaled_full_npy = np.load('test_files/expr_scaled_full_dataset.npy')
    print(omics_scaled_full_npy)
    print(omics_scaled_full_npy.shape)



if __name__ == '__main__':
    #test_dataset_preprocessor()
    #test_drug_preprocessor()
    test_omics_preprocessor()