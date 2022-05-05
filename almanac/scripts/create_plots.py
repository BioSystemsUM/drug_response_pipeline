import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_results(dl_results_file, ml_results_file, metrics, figsize=(7.5, 4.5), output_suffix=''):
	new_model_names = {'cell line (one-hot) + drugs (ECFP4)': '$cell\ line_{one\ hot} + drugs_{ECFP4}$',
				   'cell line (one-hot) + drugs (one-hot)': '$cell\ line_{one\ hot} + drugs_{one\ hot}$',
				   'Baseline (always predicts the mean of the training set)': '$baseline$',
				   'expr (landmark) + drugs (ECFP4)': '$expr_{landmark} + drugs_{ECFP4}$',
				   'expr (NCG) + drugs (ECFP4)': '$expr_{NCG} + drugs_{ECFP4}$',
				   'expr (DGI) + drugs (ECFP4)': '$expr_{DGI} + drugs_{ECFP4}$',
				   'expr (DGI + NCG) + drugs (ECFP4)': '$expr_{DGI + NCG} + drugs_{ECFP4}$',
				   'expr (COSMIC) + drugs (ECFP4)': '$expr_{COSMIC} + drugs_{ECFP4}$',
				   'expr (DGI + landmark) + drugs (ECFP4)': '$expr_{DGI + landmark} + drugs_{ECFP4}$',
				   'expr (UMAP) + drugs (ECFP4)': '$expr_{UMAP} + drugs_{ECFP4}$',
				   'expr (WGCNA) + drugs (ECFP4)': '$expr_{WGCNA} + drugs_{ECFP4}$',
				   'expr (protein coding, clustering order 1D CNN) + drugs (ECFP4)': '$expr_{protein\ coding,\ clustering\ order\ 1D\ CNN} + drugs_{ECFP4}$',
				   'expr (protein coding, chromosome position order 1D CNN) + drugs (ECFP4)': '$expr_{protein\ coding,\ chromosome\ order\ 1D\ CNN} + drugs_{ECFP4}$',
				   'expr (protein coding, chromosome position order 2D CNN) + drugs (ECFP4)': '$expr_{protein\ coding,\ chromosome\ order\ 2D\ CNN} + drugs_{ECFP4}$',
				   'expr (protein coding, clustering order 2D CNN) + drugs (ECFP4)': '$expr_{protein\ coding,\ clustering\ order\ 2D\ CNN} + drugs_{ECFP4}$',
				   'expr (protein coding) + drugs (ECFP4)': '$expr_{protein\ coding} + drugs_{ECFP4}$',
				   'expr (DGI) + drugs (MTE)': '$expr_{DGI} + drugs_{MTE}$',
				   'expr (DGI) + drugs (LayeredFP)': '$expr_{DGI} + drugs_{LayeredFP}$',
				   'expr (DGI) + drugs (GCN)': '$expr_{DGI} + drugs_{GCN}$',
				   'expr (DGI) + drugs (TextCNN)': '$expr_{DGI} + drugs_{TextCNN}$',
				   'expr (DGI) + mut (gene-level, DGI) + cnv (DGI) + drugs (ECFP4)': '$expr_{DGI} + mut_{DGI,\ gene-level} + cnv_{DGI} + drugs_{ECFP4}$',
				   'expr (DGI) + mut (pathway-level) + cnv (DGI) + drugs (ECFP4)': '$expr_{DGI} + mut_{pathway-level} + cnv_{DGI} + drugs_{ECFP4}$',
				   'expr (DGI) + drugs (one-hot)': '$expr_{DGI} + drugs_{one\ hot}$',
				   'expr (DGI) + drugs (GAT)': '$expr_{DGI} + drugs_{GAT}$',
				   'ElasticNet - expr (DGI) + drugs (ECFP4)': 'ElasticNet',
				   'XGBoost - expr (DGI) + drugs (ECFP4)': 'XGBoost',
				   'Random Forest - expr (DGI) + drugs (ECFP4)': 'Random Forest',
				   'LinearSVR - expr (DGI) + drugs (ECFP4)': 'LinearSVR',
				   'LGBM - expr (DGI) + drugs (ECFP4)': 'LGBM',
				   'Nystroem + LinearSVR - expr (DGI) + drugs (ECFP4)': 'Nystroem+LinearSVR'}

	metrics_aliases = {'pearson': 'Pearson correlation',
					   'root_mean_squared_error': 'RMSE',
					   'r2_score': 'R$^2$',
					   'spearman': 'Spearman correlation'}

	# Load results tables:
	dl_results = pd.read_csv(dl_results_file)
	dl_results.rename(columns=metrics_aliases, inplace=True)
	dl_results['model'].replace(new_model_names, inplace=True)
	ml_results = pd.read_csv(ml_results_file)
	ml_results.rename(columns=metrics_aliases, inplace=True)
	ml_results['model'].replace(new_model_names, inplace=True)

	if len(metrics) == 2:
		rows = 1
		cols = 2
	elif len(metrics) == 4:
		rows = 2
		cols = 2

	# Plots comparing different expr preprocessing steps/subnetworks
	diff_expr_models = ['$expr_{protein\ coding} + drugs_{ECFP4}$',
						'$expr_{protein\ coding,\ chromosome\ order\ 1D\ CNN} + drugs_{ECFP4}$',
						'$expr_{protein\ coding,\ clustering\ order\ 1D\ CNN} + drugs_{ECFP4}$',
						'$expr_{protein\ coding,\ chromosome\ order\ 2D\ CNN} + drugs_{ECFP4}$',
						'$expr_{protein\ coding,\ clustering\ order\ 2D\ CNN} + drugs_{ECFP4}$',
						'$expr_{landmark} + drugs_{ECFP4}$',
						'$expr_{DGI} + drugs_{ECFP4}$',
						'$expr_{COSMIC} + drugs_{ECFP4}$',
						'$expr_{NCG} + drugs_{ECFP4}$',
						'$expr_{DGI + landmark} + drugs_{ECFP4}$',
						'$expr_{DGI + NCG} + drugs_{ECFP4}$',
						'$expr_{UMAP} + drugs_{ECFP4}$',
						'$expr_{WGCNA} + drugs_{ECFP4}$',
						'$cell\ line_{one\ hot} + drugs_{ECFP4}$',
						'$cell\ line_{one\ hot} + drugs_{one\ hot}$']

	fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=400)
	if isinstance(axs[0], np.ndarray):
		axs = [ax for x in axs for ax in x]
	for i, m in enumerate(metrics):
		metric = metrics_aliases[m]
		print(metric)
		results = dl_results[dl_results['model'].isin(diff_expr_models)]
		sns.barplot(x=results['model'], y=results[metric], ax=axs[i],
					order=diff_expr_models, hue=results['model'],
					dodge=False).set(xlabel=None, ylabel=metric, title='')
	handles, labels = axs[0].get_legend_handles_labels()
	zipped = list(zip(labels, handles))
	mapping = dict(zipped)
	zipped[:] = [(x, mapping[x]) for x in diff_expr_models]
	labels, handles = map(list, zip(*zipped))
	lgd = fig.legend(handles, labels,
					 loc='upper center', bbox_to_anchor=(0.5, 0),
					 bbox_transform=plt.gcf().transFigure,
					 fancybox=True, shadow=False, ncol=2)
	for i, a in enumerate(axs):
		a.axhline(0, color='black', linewidth=0.7)
		a.get_xaxis().set_ticks([])
		a.get_legend().remove()
	plt.subplots_adjust(bottom=0.05, wspace=0.4)
	#plt.show()
	fig.savefig('../results/different_expression_models%s.svg' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
	fig.savefig('../results/different_expression_models%s.pdf' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)

	diff_drug_models = ['$expr_{DGI} + drugs_{ECFP4}$',
						'$expr_{DGI} + drugs_{LayeredFP}$',
						'$expr_{DGI} + drugs_{TextCNN}$',
						'$expr_{DGI} + drugs_{GCN}$',
						'$expr_{DGI} + drugs_{GAT}$',
						'$expr_{DGI} + drugs_{MTE}$',
						'$expr_{DGI} + drugs_{one\ hot}$',
						'$cell\ line_{one\ hot} + drugs_{one\ hot}$']
	fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=400)
	if isinstance(axs[0], np.ndarray):
		axs = [ax for x in axs for ax in x]
	for i, m in enumerate(metrics):
		metric = metrics_aliases[m]
		results = dl_results[dl_results['model'].isin(diff_drug_models)]
		sns.barplot(x=results['model'], y=results[metric], ax=axs[i],
					order=diff_drug_models, hue=results['model'],
					dodge=False).set(xlabel=None, ylabel=metric, title='')
	handles, labels = axs[0].get_legend_handles_labels()
	zipped = list(zip(labels, handles))
	mapping = dict(zipped)
	zipped[:] = [(x, mapping[x]) for x in diff_drug_models]
	labels, handles = map(list, zip(*zipped))
	lgd = fig.legend(handles, labels,
					 loc='upper center', bbox_to_anchor=(0.5, 0),
					 bbox_transform=plt.gcf().transFigure,
					 fancybox=True, shadow=False, ncol=2)
	for i, a in enumerate(axs):
		a.axhline(0, color='black', linewidth=0.7)
		a.get_xaxis().set_ticks([])
		a.get_legend().remove()
	plt.subplots_adjust(bottom=0.05, wspace=0.4)
	fig.savefig('../results/different_drug_models%s.svg' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
	fig.savefig('../results/different_drug_models%s.pdf' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)

	# Plots comparing expr+drugs and expr+mut+cnv+drugs
	extra_omics_models = ['$expr_{DGI} + drugs_{ECFP4}$',
						  '$expr_{DGI} + mut_{DGI,\ gene-level} + cnv_{DGI} + drugs_{ECFP4}$',
						  '$expr_{DGI} + mut_{pathway-level} + cnv_{DGI} + drugs_{ECFP4}$']
	fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=400)
	if isinstance(axs[0], np.ndarray):
		axs = [ax for x in axs for ax in x]
	for i, m in enumerate(metrics):
		metric = metrics_aliases[m]
		results = dl_results[dl_results['model'].isin(extra_omics_models)]
		sns.barplot(x=results['model'], y=results[metric], ax=axs[i],
					order=extra_omics_models, hue=results['model'],
					dodge=False).set(xlabel=None, ylabel=metric, title='')
	handles, labels = axs[0].get_legend_handles_labels()
	zipped = list(zip(labels, handles))
	mapping = dict(zipped)
	zipped[:] = [(x, mapping[x]) for x in extra_omics_models]
	labels, handles = map(list, zip(*zipped))
	lgd = fig.legend(handles, labels,
					 loc='upper center', bbox_to_anchor=(0.5, 0),
					 bbox_transform=plt.gcf().transFigure,
					 fancybox=True, shadow=False, ncol=1)
	for i, a in enumerate(axs):
		a.axhline(0, color='black', linewidth=0.7)
		a.get_xaxis().set_ticks([])
		a.get_legend().remove()
	plt.subplots_adjust(bottom=0.05, wspace=0.4)
	fig.savefig('../results/extra_omics_models%s.svg' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
	fig.savefig('../results/extra_omics_models%s.pdf' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)

	# Plots comparing DL and ML models
	dl_vs_ml_models = ['DL', 'ElasticNet', 'LinearSVR', 'Nystroem+LinearSVR', 'Random Forest',
					   'XGBoost', 'LGBM']
	ml_results['RMSE'] = ml_results['mean_squared_error'].apply(np.sqrt)
	ml_results = pd.concat([ml_results, dl_results[dl_results['model'] == '$expr_{DGI} + drugs_{ECFP4}$']], axis=0)
	ml_results['model'].replace(to_replace='$expr_{DGI} + drugs_{ECFP4}$', value='DL', inplace=True)
	fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=400)
	if isinstance(axs[0], np.ndarray):
		axs = [ax for x in axs for ax in x]
	for i, m in enumerate(metrics):
		metric = metrics_aliases[m]
		results = ml_results[ml_results['model'].isin(dl_vs_ml_models)]
		sns.barplot(x=results['model'], y=results[metric], ax=axs[i],
					order=dl_vs_ml_models, hue=results['model'],
					dodge=False).set(xlabel=None, ylabel=metric, title='')
	handles, labels = axs[0].get_legend_handles_labels()
	zipped = list(zip(labels, handles))
	mapping = dict(zipped)
	zipped[:] = [(x, mapping[x]) for x in dl_vs_ml_models]
	labels, handles = map(list, zip(*zipped))
	lgd = fig.legend(handles, labels,
					 loc='upper center', bbox_to_anchor=(0.5, 0),
					 bbox_transform=plt.gcf().transFigure,
					 fancybox=True, shadow=False, ncol=3)
	for i, a in enumerate(axs):
		a.axhline(0, color='black', linewidth=0.7)
		a.get_xaxis().set_ticks([])
		a.get_legend().remove()
	plt.subplots_adjust(bottom=0.05, wspace=0.4)
	fig.savefig('../results/dl_vs_ml_models%s.svg' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
	fig.savefig('../results/dl_vs_ml_models%s.pdf' % output_suffix, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)

if __name__ == '__main__':
	plot_results(dl_results_file='../results/dl_model_evaluation_results_cellminercdb_with_hyperparams.csv',
				 ml_results_file='../results/ml_model_evaluation_results_cellminercdb.csv',
				 metrics=['pearson', 'spearman', 'r2_score', 'root_mean_squared_error'],
				 figsize=(7.5, 7.5),
				 output_suffix='_recalculated_scores')
	plot_results(dl_results_file='../results/dl_model_evaluation_results_cellminercdb_with_hyperparams.csv',
				 ml_results_file='../results/ml_model_evaluation_results_cellminercdb.csv',
				 metrics=['spearman', 'r2_score'],
				 output_suffix='_paper_recalculated_scores')
