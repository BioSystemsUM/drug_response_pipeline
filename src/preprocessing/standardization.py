from chembl_structure_pipeline import standardizer
from rdkit import Chem


class SmilesStandardizer(object):
    """Applies the ChEMBL Structure Pipeline to compounds"""

    def preprocess_df(self, smiles_df, smiles_col, output_path=None):
        """
        Standardize the SMILES strings in a DataFrame.

        Parameters
        ----------
        smiles_df: DataFrame
            A DataFrame containing the molecules (in the form of SMILES strings) that will be standardized.
        smiles_col: str
            The column containing the SMILES strings.
        output_path: str or None
            Path to the file where the standardized SMILES will be saved.

        Returns
        -------
        smiles_df: DataFrame
            DataFrame with the standardized molecules.
        """
        smiles_df[smiles_col] = smiles_df[smiles_col].apply(self.preprocess_molecule)

        if output_path is not None:
            smiles_df.to_csv(output_path, index=False)

        return smiles_df

    def preprocess_molecule(self, smiles_string):
        """
        Standardizes a single molecule.

        Parameters
        ----------
        smiles_string: str
            The SMILES string representation of the molecule that will be standardized.

        Returns
        -------
        preprocessed_smiles: str
            The preprocessed molecule.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        try:
            preprocessed_mol = standardizer.standardize_mol(mol)
            preprocessed_mol, _ = standardizer.get_parent_mol(preprocessed_mol)
            preprocessed_smiles = Chem.MolToSmiles(preprocessed_mol)
        except Exception as e:
            print(e)
            preprocessed_smiles = smiles_string
        return preprocessed_smiles