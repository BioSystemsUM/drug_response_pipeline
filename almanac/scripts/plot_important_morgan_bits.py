from src.utils.utils import plot_morgan_bits


def plot_important_bits():
    """Plot the most important 'ON' bits with positive SHAP values for test set sample 11835"""
    plot_morgan_bits(smiles_file='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
                     smiles_col='SMILES_A',
                     sample_id=11835,
                     bits=[x - 1 for x in [393, 249, 923, 390, 962, 658, 328, 505, 324, 579, 707, 1011, 452, 936, 295]], # x-1 because I added 1 when naming the features
                     fp_radius=2,
                     fp_length=1024,
                     prefix='ECFP4_bit',
                     drug_img_filepath='../results/shap_analysis/sample11835_drugA.png',
                     bits_img_filepath='../results/shap_analysis/sample11835_drugA_important_bits.svg')

    plot_morgan_bits(smiles_file='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples_test.csv.gz',
                     smiles_col='SMILES_B',
                     sample_id=11835,
                     bits=[x - 1 for x in [826, 888, 689, 14, 495, 192, 795, 654, 905, 524, 466, 490, 457, 768, 203]],
                     fp_radius=2,
                     fp_length=1024,
                     prefix='ECFP4_bit',
                     drug_img_filepath='../results/shap_analysis/sample11835_drugB.png',
                     bits_img_filepath='../results/shap_analysis/sample11835_drugB_important_bits.svg')


if __name__ == '__main__':
    plot_important_bits()