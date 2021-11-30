from src.dataset.data_generator import DataGenerator

dg = DataGenerator(y_filepath='test_files/split_data_train.csv.gz', drugA_filepath='test_files/train_ECFP4_drugA.npy',
                   drugB_filepath='test_files/train_ECFP4_drugB.npy', drugA_atom_feat_filepath=None,
                   drugB_atom_feat_filepath=None, drugA_adj_filepath=None, drugB_adj_filepath=None,
                   expr_filepath='test_files/expr_scaled_train.npy', mut_filepath=None, cnv_filepath=None,
                   batch_size=32, shuffle=True)
X_batch, y_batch = dg.__getitem__(2)
print(X_batch['drugA'].shape)
X_batch, y_batch = dg.__getitem__(4)
print(X_batch['drugA'].shape)