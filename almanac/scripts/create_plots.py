import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

new_model_names = {'one-hot encoded cell line names + drug (ECFP4, 1024, dense)': '$cell\ line_{one\ hot} + drugs_{ECFP4}$',
               'one-hot encoded cell line names + one-hot encoded drugs': '$cell\ line_{one\ hot} + drugs_{one\ hot}$',
               'Baseline (always predicts the mean of the training set)': '$baseline$',
               'expr (landmark genes (REDO), dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{landmark} + drugs_{ECFP4}$',
               'expr (NCG genes, dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{NCG} + drugs_{ECFP4}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{DGI} + drugs_{ECFP4}$',
               'expr (targets full (all drug-gene interactions) + NCG genes, dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{DGI + NCG} + drugs_{ECFP4}$',
               'expr (COSMIC genes, dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{COSMIC} + drugs_{ECFP4}$',
               'expr (targets (all interactions) + landmark genes, dense, MinMaxScaler) + drug (ECFP4, Dense)': '$expr_{DGI + landmark} + drugs_{ECFP4}$',
               'expr (UMAP) + drug (ECFP4, 1024, dense)': '$expr_{UMAP} + drugs_{ECFP4}$',
               'expr (WGCNA, dense) + drug (ECFP4 1024, dense)': '$expr_{WGCNA} + drugs_{ECFP4}$',
               'expr (full, clustering order, 1D Conv) + drug (ECFP4, 1024, dense)': '$expr_{protein\ coding,\ clustering\ order\ 1D\ CNN} + drugs_{ECFP4}$',
               'expr (full, chromosome position order, 1D Conv) + drug (ECFP4, 1024, dense)': '$expr_{protein\ coding,\ chromosome\ position\ order\ 1D\ CNN} + drugs_{ECFP4}$',
               'expr (full, chromosome position order, 2D Conv) + drug (ECFP4, 1024, dense)': '$expr_{protein\ coding,\ chromosome\ position\ order\ 2D\ CNN} + drugs_{ECFP4}$',
               'expr (full, clustering order, 2D Conv) + drug (ECFP4, 1024, dense)': '$expr_{protein\ coding,\ clustering\ order\ 2D\ CNN} + drugs_{ECFP4}$',
               'expr (full, dense, MinMaxScaler) + drug (ECFP4, 1024, dense)': '$expr_{protein\ coding} + drugs_{ECFP4}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (MolecularTransformerEmbeddings, Dense)': '$expr_{DGI} + drugs_{MTE}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (LayeredFP, Dense)': '$expr_{DGI} + drugs_{LayeredFP}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (GCN)': '$expr_{DGI} + drugs_{GCN}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (TextCNN)': '$expr_{DGI} + drugs_{TextCNN}$',
               'expr (target genes, dense, MinMaxScaler) + mut (gene-level, target genes) + cnv (GISTIC, target genes) + drug (ECFP4, Dense)': '$expr_{DGI} + mut_{DGI,\ gene-level} + cnv_{DGI} + drugs_{ECFP4}$',
               'expr (target genes, dense, MinMaxScaler) + mut (pathway-level, target genes) + cnv (GISTIC, target genes) + drug (ECFP4, Dense)': '$expr_{DGI} + mut_{pathway-level} + cnv_{DGI} + drugs_{ECFP4}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + one-hot encoded drugs': '$expr_{DGI} + drugs_{one\ hot}$',
               'expr (target genes (all interactions), dense, MinMaxScaler) + drug (GAT)': '$expr_{DGI} + drugs_{GAT}$',
               'ElasticNet - expr (target genes (all interactions)) + drug (ECFP4)': 'ElasticNet',
               'XGBoost - expr (target genes (all interactions)) + drug (ECFP4)': 'XGBoost',
               'Random Forest - expr (target genes (all interactions)) + drug (ECFP4)': 'Random Forest',
               'LinearSVR - expr (target genes (all interactions)) + drug (ECFP4)': 'LinearSVR',
               'LGBM - expr (target genes (all interactions)) + drug (ECFP4)': 'LGBM',
               'Nystroem + LinearSVR - expr (target genes (all interactions)) + drug (ECFP4)': 'Nystroem+LinearSVR'}

metrics_aliases = {'keras_pearson': 'Pearson correlation',
				   'keras_r2_score': 'R$^2$',
				   'keras_spearman': 'Spearman correlation',
				   'root_mean_squared_error': 'RMSE',
				   'pearson': 'Pearson correlation',
				   'r2_score': 'R$^2$',
				   'spearman': 'Spearman correlation'}

# Load results tables:
dl_results = pd.read_csv('../results/model_evaluation_results_cellminercdb.csv')
dl_results.rename(columns=metrics_aliases, inplace=True)
dl_results['model'].replace(new_model_names, inplace=True)
ml_results = pd.read_csv('../results/ml_model_evaluation_results_cellminercdb.csv')
ml_results.rename(columns=metrics_aliases, inplace=True)
ml_results['model'].replace(new_model_names, inplace=True)

# Plots comparing different expr preprocessing steps/subnetworks
diff_expr_models = ['$expr_{protein\ coding} + drugs_{ECFP4}$',
					'$expr_{protein\ coding,\ chromosome\ position\ order\ 1D\ CNN} + drugs_{ECFP4}$',
					'$expr_{protein\ coding,\ clustering\ order\ 1D\ CNN} + drugs_{ECFP4}$',
					'$expr_{protein\ coding,\ chromosome\ position\ order\ 2D\ CNN} + drugs_{ECFP4}$',
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
                    '$cell\ line_{one\ hot} + drugs_{one\ hot}$',
                    '$baseline$']
results = dl_results[dl_results['model'].isin(diff_expr_models)]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
ax = [ax1, ax2, ax3, ax4]
for i, metric in enumerate(set(metrics_aliases.values())):
	results = dl_results[dl_results['model'].isin(diff_expr_models)]
	sns.barplot(x=results['model'], y=results[metric], ax=ax[i],
				order=diff_expr_models, hue=results['model'],
				dodge=False).set(xlabel=None, ylabel=metric, title='')
handles, labels = ax[0].get_legend_handles_labels()
zipped = list(zip(labels, handles))
mapping = dict(zipped)
zipped[:] = [(x, mapping[x]) for x in diff_expr_models]
labels, handles = map(list, zip(*zipped))
lgd = fig.legend(handles, labels,
				 loc='upper center', bbox_to_anchor=(0.5, 0),
				 bbox_transform=plt.gcf().transFigure,
				 fancybox=True, shadow=False, ncol=2, prop={'size': 16})
for i, a in enumerate(ax):
	a.axhline(0, color='black', linewidth=0.7)
	a.get_xaxis().set_ticks([])
	a.get_legend().remove()
plt.subplots_adjust(bottom=0.05, wspace=0.4)
#plt.show()
fig.savefig('../results/different_expression_models.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('../results/different_expression_models.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

diff_drug_models = ['$expr_{DGI} + drugs_{ECFP4}$',
					'$expr_{DGI} + drugs_{LayeredFP}$',
					'$expr_{DGI} + drugs_{TextCNN}$',
					'$expr_{DGI} + drugs_{GCN}$',
					'$expr_{DGI} + drugs_{GAT}$',
					'$expr_{DGI} + drugs_{MTE}$',
					'$expr_{DGI} + drugs_{one\ hot}$',
                    '$cell\ line_{one\ hot} + drugs_{one\ hot}$',
					'$baseline$']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
ax = [ax1, ax2, ax3, ax4]
for i, metric in enumerate(set(metrics_aliases.values())):
	results = dl_results[dl_results['model'].isin(diff_drug_models)]
	sns.barplot(x=results['model'], y=results[metric], ax=ax[i],
				order=diff_drug_models, hue=results['model'],
				dodge=False).set(xlabel=None, ylabel=metric, title='')
handles, labels = ax[0].get_legend_handles_labels()
zipped = list(zip(labels, handles))
mapping = dict(zipped)
zipped[:] = [(x, mapping[x]) for x in diff_drug_models]
labels, handles = map(list, zip(*zipped))
lgd = fig.legend(handles, labels,
				 loc='upper center', bbox_to_anchor=(0.5, 0),
				 bbox_transform=plt.gcf().transFigure,
				 fancybox=True, shadow=False, ncol=3, prop={'size': 16})
for i, a in enumerate(ax):
	a.axhline(0, color='black', linewidth=0.7)
	a.get_xaxis().set_ticks([])
	a.get_legend().remove()
plt.subplots_adjust(bottom=0.05, wspace=0.4)
fig.savefig('../results/different_drug_models.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('../results/different_drug_models.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


# Plots comparing expr+drugs and expr+mut+cnv+drugs
extra_omics_models = ['$expr_{DGI} + drugs_{ECFP4}$',
                      '$expr_{DGI} + mut_{DGI,\ gene-level} + cnv_{DGI} + drugs_{ECFP4}$',
					  '$expr_{DGI} + mut_{pathway-level} + cnv_{DGI} + drugs_{ECFP4}$']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
ax = [ax1, ax2, ax3, ax4]
for i, metric in enumerate(set(metrics_aliases.values())):
	results = dl_results[dl_results['model'].isin(extra_omics_models)]
	sns.barplot(x=results['model'], y=results[metric], ax=ax[i],
				order=extra_omics_models, hue=results['model'],
				dodge=False).set(xlabel=None, ylabel=metric, title='')
handles, labels = ax[0].get_legend_handles_labels()
zipped = list(zip(labels, handles))
mapping = dict(zipped)
zipped[:] = [(x, mapping[x]) for x in extra_omics_models]
labels, handles = map(list, zip(*zipped))
lgd = fig.legend(handles, labels,
				 loc='upper center', bbox_to_anchor=(0.5, 0),
				 bbox_transform=plt.gcf().transFigure,
				 fancybox=True, shadow=False, ncol=1, prop={'size': 16})
for i, a in enumerate(ax):
	a.axhline(0, color='black', linewidth=0.7)
	a.get_xaxis().set_ticks([])
	a.get_legend().remove()
plt.subplots_adjust(bottom=0.05, wspace=0.4)
fig.savefig('../results/extra_omics_models.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('../results/extra_omics_models.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

# Plots comparing DL and ML models
dl_vs_ml_models = ['DL', 'ElasticNet', 'LinearSVR', 'Nystroem+LinearSVR', 'Random Forest',
				   'XGBoost', 'LGBM']
ml_results['RMSE'] = ml_results['mean_squared_error'].apply(np.sqrt)
ml_results = pd.concat([ml_results, dl_results[dl_results['model'] == '$expr_{DGI} + drugs_{ECFP4}$']], axis=0)
print(ml_results)
ml_results['model'].replace(to_replace='$expr_{DGI} + drugs_{ECFP4}$', value='DL', inplace=True)
print(ml_results)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
ax = [ax1, ax2, ax3, ax4]
for i, metric in enumerate(set(metrics_aliases.values())):
	results = ml_results[ml_results['model'].isin(dl_vs_ml_models)]
	sns.barplot(x=results['model'], y=results[metric], ax=ax[i],
				order=dl_vs_ml_models, hue=results['model'],
				dodge=False).set(xlabel=None, ylabel=metric, title='')
handles, labels = ax[0].get_legend_handles_labels()
zipped = list(zip(labels, handles))
mapping = dict(zipped)
zipped[:] = [(x, mapping[x]) for x in dl_vs_ml_models]
labels, handles = map(list, zip(*zipped))
lgd = fig.legend(handles, labels,
				 loc='upper center', bbox_to_anchor=(0.5, 0),
				 bbox_transform=plt.gcf().transFigure,
				 fancybox=True, shadow=False, ncol=3, prop={'size': 16})
for i, a in enumerate(ax):
	a.axhline(0, color='black', linewidth=0.7)
	a.get_xaxis().set_ticks([])
	a.get_legend().remove()
plt.subplots_adjust(bottom=0.05, wspace=0.4)
fig.savefig('../results/dl_vs_ml_models.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('../results/dl_vs_ml_models.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')