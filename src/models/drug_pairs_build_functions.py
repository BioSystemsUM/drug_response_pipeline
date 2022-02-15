from inspect import getmembers

from src.models.submodels import *
from src.scoring.scoring_metrics import *
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.models import Model


DEFAULT_CHAR_DICT_STR = str({'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '/': 6, '1': 7, '2': 8, '3': 9,
                             '4': 10, '5': 11, '6': 12, '7': 13, '8': 14, '=': 15, 'C': 16, 'F': 17,
                             'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25,
                             ']': 26, '_': 27, 'c': 28, 'Cl': 29, 'Br': 30, 'n': 31, 'o': 32, 's': 33})


def expr_drug_dense_model(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA
	and drugB, with fully-connected layers in all subnetworks."""
	expr_input = Input(shape=expr_dim, name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')
	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=input_dropout,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr_drug_textcnn_model(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]',
                            predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu',
                            drug_seq_length=0, drug_n_embedding=75,
                            drug_char_dict=DEFAULT_CHAR_DICT_STR,
                            drug_kernel_sizes='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
                            drug_num_filters='[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
                            drug_dropout=0.25, drug_l1=0, drug_l2=0, l1=0, l2=0,
                            input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA and
	drugB. The expression subnetwork uses fully connected layers, and the drug subnetworks are based on the TextCNN
	model."""
	expr_input = Input(shape=expr_dim, name='expr')
	drug1_input = Input(shape=(drug_seq_length,), dtype=tf.int32, name='drugA')
	drug2_input = Input(shape=(drug_seq_length,), dtype=tf.int32, name='drugB')
	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout, )
	drug_submodel = textcnn_submodel(seq_length=drug_seq_length, n_embedding=drug_n_embedding,
	                                 char_dict=drug_char_dict, kernel_sizes=drug_kernel_sizes,
	                                 num_filters=drug_num_filters, dropout=drug_dropout, l1=drug_l1, l2=drug_l2)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr1dconv_drug_dense_model(expr_dim=None, drug_dim=None, expr_num_filters='[32, 32]', expr_kernel_sizes='[2, 2]',
                                expr_pool_size=2, expr_batchnorm=True, drug_hlayers_sizes='[10]',
                                predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                                l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA and
	drugB. The expression subnetwork is a 1D CNN and the drug subnetworks use fully-connected layers."""
	expr_input = Input(shape=(expr_dim, 1), name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	expr = conv1d_submodel(expr_input, num_filters=expr_num_filters, kernel_sizes=expr_kernel_sizes,
	                       pool_size=expr_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=expr_batchnorm,
	                       hidden_activation=hidden_activation)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr1dconv_drug_textcnn_model(expr_dim=None, drug_dim=None, expr_num_filters='[32, 32]', expr_kernel_sizes='[2, 2]',
                                  expr_pool_size=2, expr_batchnorm=True,
                                  predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu',
                                  drug_seq_length=0, drug_n_embedding=75,
                                  drug_char_dict=DEFAULT_CHAR_DICT_STR,
                                  drug_kernel_sizes='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
                                  drug_num_filters='[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
                                  drug_dropout=0.25, drug_l1=0, drug_l2=0, l1=0, l2=0,
                                  input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA and
	drugB. The expression subnetwork is a 1D CNN and the drug subnetworks are TextCNN submodels."""
	expr_input = Input(shape=(expr_dim, 1), name='expr')
	drug1_input = Input(shape=(drug_seq_length,), dtype=tf.int32, name='drugA')
	drug2_input = Input(shape=(drug_seq_length,), dtype=tf.int32, name='drugB')

	expr = conv1d_submodel(expr_input, num_filters=expr_num_filters, kernel_sizes=expr_kernel_sizes,
	                       pool_size=expr_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=expr_batchnorm,
	                       hidden_activation=hidden_activation)
	drug_submodel = textcnn_submodel(seq_length=drug_seq_length, n_embedding=drug_n_embedding,
	                                 char_dict=drug_char_dict, kernel_sizes=drug_kernel_sizes,
	                                 num_filters=drug_num_filters, dropout=drug_dropout, l1=drug_l1, l2=drug_l2)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr2dconv_drug_dense_model(expr_dim=None, drug_dim=None, expr_num_filters='[32, 32]',
                                expr_kernel_size=(3, 3), expr_kernel_size_rest=(3, 3),
                                expr_pool_size=(2, 2), expr_batchnorm=True, drug_hlayers_sizes='[10]',
                                predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                                l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA and
	drugB. The expression subnetwork is a 2D CNN and the drug subnetworks use fully-connected layers."""
	expr_input = Input(shape=expr_dim, name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	expr = conv2d_submodel(expr_input, num_filters=expr_num_filters, kernel_size=expr_kernel_size,
	                       pool_size=expr_pool_size, initializer=initializer, batchnorm=expr_batchnorm,
	                       hidden_activation=hidden_activation)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr2dconv_densenet_drug_dense_model(expr_dim=None, drug_dim=None, drug_hlayers_sizes='[10]',
                                         predictor_hlayers_sizes='[10]', initializer='he_normal',
                                         hidden_activation='relu', l1=0, l2=0, input_dropout=0, hidden_dropout=0,
                                         optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA and
	drugB. The expression subnetwork is a 2D CNN (Densenet) and the drug subnetworks use fully-connected layers."""
	expr_input = Input(shape=expr_dim, name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	expr = densenet_submodel(expr_input)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def mut_cnv_drug_dense_model(mut_dim=None, cnv_dim=None, drug_dim=None, mut_hlayers_sizes='[10]',
                             cnv_hlayers_sizes='[10]', drug_hlayers_sizes='[10]', predictor_hlayers_sizes='[10]',
                             initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                             hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for mutation data, CNV data,
	drugA and drugB. All subnetworks use fully-connected layers."""
	mut_input = Input(shape=mut_dim, name='mut')
	cnv_input = Input(shape=cnv_dim, name='cnv')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	mut = dense_submodel(mut_input, hlayers_sizes=mut_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	cnv = dense_submodel(cnv_input, hlayers_sizes=cnv_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([mut, cnv, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[mut_input, cnv_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr_mut_cnv_drug_dense_model(expr_dim=None, mut_dim=None, cnv_dim=None, drug_dim=None, expr_hlayers_sizes='[10]',
                                  mut_hlayers_sizes='[10]', cnv_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                                  predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu',
                                  l1=0, l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, mutation
	data, CNV data, drugA and drugB. All subnetworks use fully-connected layers."""
	expr_input = Input(shape=expr_dim, name='expr')
	mut_input = Input(shape=mut_dim, name='mut')
	cnv_input = Input(shape=cnv_dim, name='cnv')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	mut = dense_submodel(mut_input, hlayers_sizes=mut_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	cnv = dense_submodel(cnv_input, hlayers_sizes=cnv_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, mut, cnv, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[mut_input, cnv_input, expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr_mut_cnv_drug_gcn_model(expr_dim=None, mut_dim=None, cnv_dim=None, drug_dim=None, expr_hlayers_sizes='[10]',
                                mut_hlayers_sizes='[10]', cnv_hlayers_sizes='[10]', drug_n_atom_features=30,
                                drug_gcn_layers='[64, 64]', drug_residual_connection=True, drug_dropout=0.5,
                                predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu',
                                l1=0, l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, mutation
	data, CNV data, drugA and drugB. The expression, mutation and CNV subnetworks use fully-connected layers and the
	drug subnetworks use GCN."""
	expr_input = Input(shape=expr_dim, name='expr')
	mut_input = Input(shape=mut_dim, name='mut')
	cnv_input = Input(shape=cnv_dim, name='cnv')
	drugA_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')
	drugB_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugB_atom_feat')
	drugB_adjacency_input = Input(shape=(None, None), name='drugB_adj')

	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	mut = dense_submodel(mut_input, hlayers_sizes=mut_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	cnv = dense_submodel(cnv_input, hlayers_sizes=cnv_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	drug_submodel = gcn_submodel(drug_n_atom_features, gcn_layers=drug_gcn_layers, residual=drug_residual_connection,
	                             dropout_rate=drug_dropout, l2=l2)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])
	drugB = drug_submodel([drugB_nodes_input, drugB_adjacency_input])

	concat = concatenate([expr, mut, cnv, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, mut_input, cnv_input, drugA_nodes_input, drugA_adjacency_input,
	                      drugB_nodes_input, drugB_adjacency_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def mut_cnv_drug_gcn_model(mut_dim=None, cnv_dim=None, drug_dim=None, mut_hlayers_sizes='[10]',
                           cnv_hlayers_sizes='[10]', drug_n_atom_features=30, drug_gcn_layers='[64, 64]',
                           drug_residual_connection=True, drug_dropout=0.5, predictor_hlayers_sizes='[10]',
                           initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                           hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for mutation data, CNV data,
	drugA and drugB. The mutation and CNV subnetworks use fully-connected layers and the drug subnetworks use GCN."""
	mut_input = Input(shape=mut_dim, name='mut')
	cnv_input = Input(shape=cnv_dim, name='cnv')
	drugA_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')
	drugB_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugB_atom_feat')
	drugB_adjacency_input = Input(shape=(None, None), name='drugB_adj')

	mut = dense_submodel(mut_input, hlayers_sizes=mut_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	cnv = dense_submodel(cnv_input, hlayers_sizes=cnv_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                     hidden_activation=hidden_activation, input_dropout=input_dropout,
	                     hidden_dropout=hidden_dropout)
	drug_submodel = gcn_submodel(drug_n_atom_features, gcn_layers=drug_gcn_layers, residual=drug_residual_connection,
	                             dropout_rate=drug_dropout, l2=l2)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])
	drugB = drug_submodel([drugB_nodes_input, drugB_adjacency_input])

	concat = concatenate([mut, cnv, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[mut_input, cnv_input, drugA_nodes_input, drugA_adjacency_input,
	                      drugB_nodes_input, drugB_adjacency_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def mut1dconv_cnv1dconv_drug_dense_model(mut_dim=None, cnv_dim=None, drug_dim=None, mut_num_filters='[32, 32]',
                                         mut_kernel_sizes='[2, 2]', mut_pool_size=2, mut_batchnorm=True,
                                         cnv_num_filters='[32, 32]', cnv_kernel_sizes='[2, 2]', cnv_pool_size=2,
                                         cnv_batchnorm=True, drug_hlayers_sizes='[10]', predictor_hlayers_sizes='[10]',
                                         initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                                         hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for mutation data, CNV data,
	drugA and drugB. The mutation and CNV subnetworks are 1D CNNs and the drug subnetworks use fully-connected layers."""
	mut_input = Input(shape=(mut_dim, 1), name='mut')
	cnv_input = Input(shape=(cnv_dim, 1), name='cnv')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	mut = conv1d_submodel(mut_input, num_filters=mut_num_filters, kernel_sizes=mut_kernel_sizes,
	                      pool_size=mut_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=mut_batchnorm,
	                      hidden_activation=hidden_activation)
	cnv = conv1d_submodel(cnv_input, num_filters=cnv_num_filters, kernel_sizes=cnv_kernel_sizes,
	                      pool_size=cnv_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=cnv_batchnorm,
	                      hidden_activation=hidden_activation)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([mut, cnv, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[mut_input, cnv_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def mut1dconv_cnv1dconv_expr1dconv_drug_dense_model(mut_dim=None, cnv_dim=None, expr_dim=None, drug_dim=None,
                                                    mut_num_filters='[32, 32]', mut_kernel_sizes='[2, 2]',
                                                    mut_pool_size=2, mut_batchnorm=True, cnv_num_filters='[32, 32]',
                                                    cnv_kernel_sizes='[2, 2]', cnv_pool_size=2, cnv_batchnorm=True,
                                                    expr_num_filters='[32, 32]', expr_kernel_sizes='[2, 2]',
                                                    expr_pool_size=2, expr_batchnorm=True, drug_hlayers_sizes='[10]',
                                                    predictor_hlayers_sizes='[10]', initializer='he_normal',
                                                    hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                                                    hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data,
	mutation data, CNV data, drugA and drugB. The expression, mutation and CNV subnetworks are 1D CNNs and the drug
	subnetworks use fully-connected layers."""
	mut_input = Input(shape=(mut_dim, 1), name='mut')
	cnv_input = Input(shape=(cnv_dim, 1), name='cnv')
	expr_input = Input(shape=(expr_dim, 1), name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')

	mut = conv1d_submodel(mut_input, num_filters=mut_num_filters, kernel_sizes=mut_kernel_sizes,
	                      pool_size=mut_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=mut_batchnorm,
	                      hidden_activation=hidden_activation)
	cnv = conv1d_submodel(cnv_input, num_filters=cnv_num_filters, kernel_sizes=cnv_kernel_sizes,
	                      pool_size=cnv_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=cnv_batchnorm,
	                      hidden_activation=hidden_activation)
	expr = conv1d_submodel(expr_input, num_filters=expr_num_filters, kernel_sizes=expr_kernel_sizes,
	                       pool_size=expr_pool_size, initializer=initializer, l1=l1, l2=l2, batchnorm=expr_batchnorm,
	                       hidden_activation=hidden_activation)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([mut, cnv, expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[mut_input, cnv_input, expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def two_drug_gcn_model(n_atom_features=30, gcn_layers='[64, 64]', residual=True, activation='relu', dropout=0.5,
                       predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                       l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding GCN subnetworks for drugA and drugB."""
	# drugA_nodes_input = Input(shape=(max_n_atoms, drug_n_atom_features), name='drugA_atom_feat')
	# drugA_adjacency_input = Input(shape=(max_n_atoms, max_n_atoms), name='drugA_adj')
	# drugB_nodes_input = Input(shape=(max_n_atoms, drug_n_atom_features), name='drugB_atom_feat')
	# drugB_adjacency_input = Input(shape=(max_n_atoms, max_n_atoms), name='drugB_adj')
	drugA_nodes_input = Input(shape=(None, n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')
	drugB_nodes_input = Input(shape=(None, n_atom_features), name='drugB_atom_feat')
	drugB_adjacency_input = Input(shape=(None, None), name='drugB_adj')
	drug_submodel = gcn_submodel(n_atom_features, gcn_layers=gcn_layers, residual=residual,
	                             activation=activation, dropout_rate=dropout, l2=l2)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])
	drugB = drug_submodel([drugB_nodes_input, drugB_adjacency_input])

	concat = concatenate([drugA, drugB])

	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=input_dropout,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[drugA_nodes_input, drugA_adjacency_input, drugB_nodes_input, drugB_adjacency_input],
	              outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr_drug_gcn_model(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_n_atom_features=30,
                        drug_gcn_layers='[64, 64]', drug_residual_connection=True, drug_dropout=0.5,
                        predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0, l2=0,
                        input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA
	and drugB. The expression subnetwork uses fully-connected layers and the drug subnetworks are GCNs"""
	expr_input = Input(shape=expr_dim, name='expr')
	drugA_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')
	drugB_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugB_atom_feat')
	drugB_adjacency_input = Input(shape=(None, None), name='drugB_adj')

	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	drug_submodel = gcn_submodel(drug_n_atom_features, gcn_layers=drug_gcn_layers, residual=drug_residual_connection,
	                             dropout_rate=drug_dropout, l2=l2)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])
	drugB = drug_submodel([drugB_nodes_input, drugB_adjacency_input])

	concat = concatenate([expr, drugA, drugB])

	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=input_dropout,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(
		inputs=[expr_input, drugA_nodes_input, drugA_adjacency_input, drugB_nodes_input, drugB_adjacency_input],
		outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


def expr_drug_gat_model(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_n_atom_features=30,
                        drug_gat_layers='[64, 64]', drug_num_attention_heads=8, drug_concat_heads=True,
                        drug_residual_connection=True, drug_dropout=0.5, predictor_hlayers_sizes='[10]',
                        initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                        hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA
	and drugB. The expression subnetwork uses fully-connected layers and the drug subnetworks are GATs"""
	expr_input = Input(shape=expr_dim, name='expr')
	drugA_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')
	drugB_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugB_atom_feat')
	drugB_adjacency_input = Input(shape=(None, None), name='drugB_adj')

	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	drug_submodel = gat_submodel(n_atom_features=drug_n_atom_features, gat_layers=drug_gat_layers,
	                             n_attention_heads=drug_num_attention_heads, concat_heads=drug_concat_heads,
	                             residual=drug_residual_connection, dropout_rate=drug_dropout)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])
	drugB = drug_submodel([drugB_nodes_input, drugB_adjacency_input])

	concat = concatenate([expr, drugA, drugB])

	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=input_dropout,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(
		inputs=[expr_input, drugA_nodes_input, drugA_adjacency_input, drugB_nodes_input, drugB_adjacency_input],
		outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[MeanSquaredError(), RootMeanSquaredError(), keras_r2_score, keras_spearman, keras_pearson])

	return model


if __name__ == '__main__':
	from tensorflow.keras.utils import plot_model

	# m1 = expr_drug_dense_model(expr_dim=10, drug_dim=10)
	# print(m1.metrics)
	# print(m1.summary())
	# plot_model(m1, to_file='model.png')
	#
	# m2 = mut_cnv_expr_drug_dense_model(mut_dim=10, cnv_dim=10, expr_dim=10, drug_dim=10)
	# print(m2.metrics)
	# print(m2.summary())
	# plot_model(m2, to_file='mut_cnv_expr_drugs_model.png')

	# m3 = two_drug_gcn_model()
	# # print(m3.metrics)
	# print(m3.summary())
	# plot_model(m3, to_file='two_drug_gcn_model.png')
	# node_feats_a = np.load('smiles_a_nodes_test.npy')
	# adj_feats_a = np.load('smiles_a_adj_mat_test.npy')
	# node_feats_b = np.load('smiles_b_nodes_test.npy')
	# adj_feats_b = np.load('smiles_b_adj_mat_test.npy')
	# response_df = pd.read_csv('gcn_test_response_data.csv')
	# y = response_df['COMBOSCORE'].values
	# input_dict = {'drugA_atom_feat': node_feats_a,
	#               'drugA_adj': adj_feats_a,
	#               'drugB_atom_feat': node_feats_b,
	#               'drugB_adj': adj_feats_b}
	# m3.fit(x=input_dict, y=y, epochs=2)

	m4 = expr_drug_gcn_model(expr_dim=10)
	print(m4.summary())
	plot_model(m4, to_file='expr_drug_gcn_model.png')
