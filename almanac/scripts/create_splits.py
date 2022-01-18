import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold


def create_LDCO_splits_cellminercdb(random_seed, output_path):
    """Creates train/validation/test sets (80%/10%/10% split), where each set has different drug combinations not included in the other sets"""
    df = pd.read_csv(
        '../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv')
    gkf = GroupKFold(n_splits=10)
    np.random.seed(seed=random_seed)
    groups = df['GROUP'].tolist()
    np.random.shuffle(groups)
    split_inds_folds = list(gkf.split(X=df, groups=groups))
    test_inds = split_inds_folds[0][1].tolist()
    val_inds = split_inds_folds[1][1].tolist()
    train_inds = []
    for i in range(2, 10):
        train_inds.extend(split_inds_folds[i][1].tolist())
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
    create_LDCO_splits_cellminercdb(random_seed=12321,
                                    output_path='../data/splits/train_val_test_groups_split_inds_12321.pkl')
