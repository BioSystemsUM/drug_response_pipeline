import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale, maxabs_scale, minmax_scale, robust_scale, quantile_transform, power_transform
from rdkit.Chem import MolFromSmiles, SanitizeMol


# analyze features and y - values distribution, missing values, make plots, etc.

def plot_expr_distribution(expr_filepath, plot_file):
	'''Plot the distribution of the expression level of each gene'''
	df = pd.read_csv(expr_filepath)
	df.set_index('cell_line_name', inplace=True)

	# distribution of values per column
	cols = df.columns.tolist()[0:11]  #
	for col in cols:
		plt.hist(df[col].values, bins=10)
		fig_path1 = col + '_original_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		# Standard scaling
		plt.hist(scale(df[col].values), bins=10)
		fig_path1 = col + '_standardized_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(minmax_scale(df[col].values), bins=10)
		fig_path1 = col + '_minmax_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(maxabs_scale(df[col].values), bins=10)
		fig_path1 = col + '_maxabs_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(robust_scale(df[col].values), bins=10)
		fig_path1 = col + '_robustscaler_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(quantile_transform(df[col].values.reshape(-1, 1), output_distribution='normal'), bins=10)
		fig_path1 = col + '_quantiletransform_normal_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(quantile_transform(df[col].values.reshape(-1, 1), output_distribution='uniform'), bins=10)
		fig_path1 = col + '_quantiletransform_uniform_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()

		plt.hist(power_transform(df[col].values.reshape(-1, 1)), bins=10)
		fig_path1 = col + '_powertransform_yeojohnson_' + plot_file
		plt.savefig(fig_path1, format='png')
		plt.clf()


def check_cell_line_existence(omics_dataset_path):
	drugcomb_cell_lines = pd.read_csv('../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles.csv')[
		'cell_line_name'].unique().tolist()
	omics_cell_lines = pd.read_csv(omics_dataset_path)['cell_line_name']
	if 'gistic' in omics_dataset_path:
		omics_cell_lines = [x.replace('_', '-') for x in omics_cell_lines]

	print('In drugcomb but not in omics dataset:')
	print(sorted(list(set(drugcomb_cell_lines).difference(set(omics_cell_lines)))))
	print('In omics dataset but not in drugcomb:')
	print(sorted(list(set(omics_cell_lines).difference(set(drugcomb_cell_lines)))))

	# mut:
	# In drugcomb but not in omics dataset:
	# ['A549', 'HCT116', 'MDA-MB-468', 'NCIH23', 'OVCAR3', 'UACC62']
	# In CellminerCDB but not in drugcomb:
	# ['A549/ATCC', 'HCT-116', 'MDA-N', 'NCI-H23', 'OVCAR-3', 'UACC-62']

	# expr:
	# In drugcomb but not in omics dataset:
	# ['A549', 'HCT116', 'MDA-MB-468', 'NCIH23', 'OVCAR3', 'UACC62']
	# In omics dataset but not in drugcomb:
	# ['A549/ATCC', 'HCT-116', 'MDA-N', 'NCI-H23', 'OVCAR-3', 'UACC-62']

	# gistic file from cBioportal:
	# In drugcomb but not in omics dataset:
	# ['786-0', 'BT-549', 'CAKI-1', 'CCRF-CEM', 'COLO 205', 'DU-145', 'HCC-2998', 'HCT-15', 'HCT116', 'HL-60(TB)', 'HOP-62', 'HOP-92', 'HS 578T', 'K-562', 'LOX IMVI', 'MALME-3M', 'MDA-MB-231', 'MDA-MB-435', 'MDA-MB-468', 'MOLT-4', 'NCI-H226', 'NCI-H322M', 'NCI-H460', 'NCI-H522', 'NCI/ADR-RES', 'NCIH23', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'OVCAR3', 'PC-3', 'RPMI-8226', 'RXF 393', 'SF-268', 'SF-295', 'SF-539', 'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'SK-OV-3', 'SNB-19', 'SNB-75', 'SW-620', 'T-47D', 'TK-10', 'U251', 'UACC-257', 'UACC62', 'UO-31']
	# In omics dataset but not in drugcomb:
	# ['BT_549', 'CAKI_1', 'CCRF_CEM', 'COLO205', 'CSF_268', 'CSF_295', 'CSF_539', 'CSNB_19', 'CSNB_75', 'CU251', 'DU_145', 'HCC_2998', 'HCT_116', 'HCT_15', 'HL_60', 'HOP_62', 'HOP_92', 'HS578T', 'K_562', 'LOXIMVI', 'MALME_3M', 'MDA_MB_231', 'MDA_MB_435', 'MDA_N', 'MOLT_4', 'NCI_ADR_RES', 'NCI_H226', 'NCI_H23', 'NCI_H322M', 'NCI_H460', 'NCI_H522', 'OVCAR_3', 'OVCAR_4', 'OVCAR_5', 'OVCAR_8', 'PC_3', 'RPMI_8226', 'RXF_393', 'SK_MEL_2', 'SK_MEL_28', 'SK_MEL_5', 'SK_OV_3', 'SW_620', 'T47D', 'TK_10', 'UACC_257', 'UACC_62', 'UO_31', 'X786_0']

	# gistic file from cBioportal after replacing '_' with '-':
	# ['786-0', 'COLO 205', 'HCT116', 'HL-60(TB)', 'HS 578T', 'LOX IMVI', 'MDA-MB-468', 'NCI/ADR-RES', 'NCIH23', 'OVCAR3', 'RXF 393', 'SF-268', 'SF-295', 'SF-539', 'SNB-19', 'SNB-75', 'T-47D', 'U251', 'UACC62']
	# In omics dataset but not in drugcomb:
	# ['COLO205', 'CSF-268', 'CSF-295', 'CSF-539', 'CSNB-19', 'CSNB-75', 'CU251', 'HCT-116', 'HL-60', 'HS578T', 'LOXIMVI', 'MDA-N', 'NCI-ADR-RES', 'NCI-H23', 'OVCAR-3', 'RXF-393', 'T47D', 'UACC-62', 'X786-0']


def check_smiles_rdkit():
	'''Check if all SMILES are read by RDKit'''
	response_df = pd.read_csv('../data/nci_almanac_preprocessed/response/DataTable_ALMANAC_with_smiles.csv')
	smiles = response_df['smiles_row'].unique().tolist() + response_df['smiles_col'].unique().tolist()
	for smi in smiles:
		m = MolFromSmiles(smi, sanitize=False)
		if m is None:
			print('invalid SMILES')
			print(smi)
		else:
			try:
				SanitizeMol(m)
			except:
				print('invalid chemistry')
				print(smi)
	# this prints out nothing, so all RDKit should be able to process all of these SMILES



if __name__ == '__main__':
	plot_expr_distribution('../data/nci_almanac_preprocessed/omics/rnaseq_fpkm_prot_coding.csv',
	                       'rnaseq_fpkm_hist.png')
	#check_cell_line_existence('../data/nci_almanac_preprocessed/omics/cnvs_gistic_prot_coding.csv')
	# check_smiles_rdkit()
