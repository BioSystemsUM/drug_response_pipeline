from chembl_structure_pipeline import standardizer
from rdkit import Chem


class SmilesStandardizer(object):

    def preprocess_df(self, smiles_df, smiles_col, output_path=None):
        smiles_df[smiles_col] = smiles_df[smiles_col].apply(self.preprocess_molecule)

        if output_path is not None:
            smiles_df.to_csv(output_path, index=False)

        return smiles_df

    def preprocess_molecule(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        try:
            preprocessed_mol = standardizer.standardize_mol(mol)
            preprocessed_mol, _ = standardizer.get_parent_mol(preprocessed_mol)
            preprocessed_smiles = Chem.MolToSmiles(preprocessed_mol)
        except Exception as e:
            print(e)
            preprocessed_smiles = smiles_string
        return preprocessed_smiles