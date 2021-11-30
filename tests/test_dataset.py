from copy import copy
from src.dataset.dataset import MultiInputDataset


def test_multiinputdataset():
    dataset = MultiInputDataset(response_dataset_path='test_files/split_data_train.csv.gz',
                                id_cols=['NSC1', 'NSC2', 'CELLNAME'],
                                output_col='COMBOSCORE',
                                input_order=['expr', 'drugA', 'drugB'])
    print(dataset.response_dataset)
    print(dataset.y)
    dataset.save_y('test_files/y_train.pkl')

    print(dataset.get_n_rows())

    print(dataset.get_row_ids())

    dataset.load_expr(rnaseq_file='test_files/expr_scaled_train.npy')
    dataset.load_drugA(drugA_file='test_files/train_ECFP4_drugA.npy')
    dataset.load_drugB(drugB_file='test_files/train_ECFP4_drugB.npy')
    print(dataset.X_dict)
    dataset.save_dataset_dict('test_files/X_dict_train.pkl')
    print(dataset.get_dataset_dimensions(model_type='expr_drug_dense_model'))

    dataset.X_dict_to_list()
    print(dataset.X_list)
    print(len(dataset.X_list))

    dataset.concat_features()
    print(dataset.X)
    print(type(dataset.X))
    print(dataset.X.shape)

    #dataset.load_feature_names(feature_names_filepath='')

    selected = dataset.select(indices=[0, 1, 2])
    print(selected.get_n_rows())
    print(selected.X_dict)
    print(selected.y)
    print(selected.response_dataset)

    sampled_dataset = dataset.sample(n=5)
    print(sampled_dataset.get_n_rows())
    print(sampled_dataset.X_dict)
    print(sampled_dataset.y)
    print(sampled_dataset.response_dataset)

    dataset2 = copy(dataset)
    print(dataset2.get_n_rows())
    concat_dataset = dataset.concat_datasets(dataset_to_concat=dataset2)
    print(concat_dataset.get_n_rows())
    print(concat_dataset.X_dict)
    print(concat_dataset.y)
    print(concat_dataset.response_dataset)


def test_loading_dataset_from_files():
    dataset = MultiInputDataset()
    dataset.load_dataset_dict('test_files/X_dict_train.pkl')
    dataset.load_y('test_files/y_train.pkl')
    print(dataset.X_dict)
    print(dataset.y)


if __name__ == '__main__':
    test_multiinputdataset()
    test_loading_dataset_from_files()