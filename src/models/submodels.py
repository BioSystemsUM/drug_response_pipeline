from ast import literal_eval

from deepchem.models.layers import Highway, DTNNEmbedding
from spektral.layers import GCNConv, GlobalSumPool, GATConv
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, \
	AlphaDropout, Conv1D, Lambda, Concatenate, MaxPooling1D, GlobalMaxPooling1D, Multiply, \
	Conv2D, MaxPooling2D, GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU


def drug_dense_submodel(input_dim, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                        hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""Build a dense (fully-connected) submodel for drugs, so that it can be used later on to share weights
	between two drug subnetworks."""
	input_layer = Input(shape=input_dim)

	output = dense_submodel(input_layer, hlayers_sizes, l1_regularization, l2_regularization,
	                        hidden_activation, input_dropout, hidden_dropout)

	submodel = Model(inputs=input_layer, outputs=output, name='drug_dense_submodel')

	return submodel


def dense_submodel(input_layer, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                   hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""Build a dense (fully-connected) subnetwork."""
	hlayers_sizes = literal_eval(hlayers_sizes)  # because hlayers_sizes was passed as a string

	if hidden_activation == 'selu':
		# selu must be used with the LecunNormal initializer and AlphaDropout instead of normal Dropout
		initializer = 'lecun_normal'
		dropout = AlphaDropout
		batchnorm = False
	else:
		initializer = 'he_normal'
		dropout = Dropout
		batchnorm = True

	if input_dropout > 0:
		x = dropout(rate=input_dropout)(input_layer)
	else:
		x = input_layer

	for i in range(len(hlayers_sizes)):
		x = Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
		          kernel_regularizer=l1_l2(l1=l1_regularization, l2=l2_regularization))(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)

		if batchnorm:
			x = BatchNormalization()(x)

		if hidden_dropout > 0:
			x = dropout(rate=hidden_dropout)(x)

	return x


def textcnn_submodel(seq_length, char_dict, n_embedding=75,
                     kernel_sizes='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]',
                     num_filters='[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]',
                     l1=0, l2=0, dropout=0.25):
	"""Build a TextCNN subnetwork.

	Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/models/text_cnn.py
	"""
	input_layer = Input(shape=(seq_length,), dtype=tf.int32, name='drug')

	kernel_sizes = literal_eval(kernel_sizes)
	num_filters = literal_eval(num_filters)
	kernel_regularizer = l1_l2(l1=l1, l2=l2)
	# char_dict = dict(char_dict)
	char_dict = literal_eval(char_dict)

	embedding = DTNNEmbedding(
		n_embedding=n_embedding,
		periodic_table_length=len(char_dict.keys()) + 1)(input_layer)

	pooled_outputs = []
	conv_layers = []
	for filter_size, num_filter in zip(kernel_sizes, num_filters):
		# Multiple convolutional layers with different filter widths
		conv_layers.append(
			Conv1D(kernel_size=filter_size, filters=num_filter, padding='valid',
			       kernel_regularizer=kernel_regularizer)(embedding))
		# Max-over-time pooling
		reduced = Lambda(lambda x: math.reduce_max(x, axis=1))(conv_layers[-1])
		pooled_outputs.append(reduced)
	# Concat features from all filters(one feature per filter)
	concat_outputs = Concatenate(axis=1)(pooled_outputs)
	dropout = Dropout(rate=dropout)(concat_outputs)
	dense = Dense(200, activation='relu')(dropout)
	# Highway layer from https://arxiv.org/pdf/1505.00387.pdf
	gather = Highway()(dense)

	submodel = Model(inputs=input_layer, outputs=gather, name='drug_textcnn_submodel')

	return submodel


def conv1d_submodel(input_layer, num_filters='[32, 32]', kernel_sizes='[2, 2]', pool_size=2,
                    initializer='glorot_uniform', hidden_activation='relu', l1=0, l2=0, batchnorm=True):
	"""Build a 1D CNN subnetwork."""
	kernel_sizes = literal_eval(kernel_sizes)
	num_filters = literal_eval(num_filters)
	kernel_regularizer = l1_l2(l1=l1, l2=l2)

	x = input_layer

	for i in range(len(num_filters)):
		x = Conv1D(filters=num_filters[i], kernel_size=kernel_sizes[i], padding='valid',
		           kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, strides=1)(x)
		# if batchnorm:
		#     x = BatchNormalization()(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)
		if batchnorm:
			x = BatchNormalization()(x)
		if i == len(num_filters) - 1:
			x = GlobalMaxPooling1D()(x)
		else:
			x = MaxPooling1D(pool_size=pool_size)(x)

	return x


def conv2d_submodel(input_layer, num_filters='[32, 32]', kernel_size=(3, 3), pool_size=(2, 2),
                    initializer='glorot_uniform', hidden_activation='relu', batchnorm=True):
	"""Build a 2D CNN subnetwork."""
	num_filters = literal_eval(num_filters)

	x = input_layer

	for i in range(len(num_filters)):
		x = Conv2D(filters=num_filters[i], kernel_size=kernel_size, kernel_initializer=initializer)(x)
		# if batchnorm:
		#     x = BatchNormalization()(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)
		if batchnorm:
			x = BatchNormalization()(x)
		if i == len(num_filters) - 1:
			x = GlobalAveragePooling2D()(x)
		else:
			x = MaxPooling2D(pool_size=pool_size)(x)

	return x


def densenet_submodel(input_layer):
	"""Build a DenseNet submodel."""
	model = DenseNet121(include_top=False, input_tensor=input_layer, pooling='avg')

	x = model(input_layer)

	return x


def gcn_submodel(n_atom_features, gcn_layers='[64, 64]', residual=False, activation='relu',
                 dropout_rate=0.5, l2=0):
	"""Build a Graph Convolutional Network (GCN) (Kipf et al, 2017) submodel"""
	# I'm using a different kernel initializer though
	# GlobalSumPool was not part of the original GCN model, but is necessary to get a graph-level embedding
	gcn_layers = literal_eval(gcn_layers)
	regularizer = l1_l2(l1=0, l2=l2)
	# nodes_input = Input(shape=(max_n_atoms, drug_n_atom_features))
	# adjacency_input = Input(shape=(max_n_atoms, max_n_atoms))
	nodes_input = Input(shape=(None, n_atom_features))
	adjacency_input = Input(shape=(None, None))
	node_feats = nodes_input
	for n_channels in gcn_layers:
		x = GCNConv(n_channels, activation, kernel_initializer='he_normal',
		            kernel_regularizer=regularizer)([node_feats, adjacency_input])
		if residual:  # add a drug_residual_connection connection (as implemented in GCNPredictor from DGL LifeSci)
			res_feats = Dense(n_channels, activation=activation)(
				node_feats)  # the DGL LifeSci code does this first. I think it's needed because the shape of the original inputs is not the same as the outputs of GCNConv
			x = add([x, res_feats])  # drug_residual_connection connection
		x = Dropout(dropout_rate)(x)
		node_feats = x
	x = GlobalSumPool()(x)

	submodel = Model(inputs=[nodes_input, adjacency_input], outputs=[x], name='drug_gcn_submodel')

	return submodel


def gat_submodel(n_atom_features, gat_layers='[64, 64]', n_attention_heads=8, concat_heads=True, residual=False,
                 dropout_rate=0.5, l2=0):
	"""Build a Graph Attention Network (GAT) (Velickovic et al, 2018) submodel."""
	gat_layers = literal_eval(gat_layers)
	regularizer = l1_l2(l1=0, l2=l2)
	nodes_input = Input(shape=(None, n_atom_features))
	adjacency_input = Input(shape=(None, None))
	node_feats = nodes_input
	for n_channels in gat_layers:
		x = GATConv(n_channels, activation='elu', attn_heads=n_attention_heads, concat_heads=concat_heads,
		            dropout_rate=dropout_rate, kernel_initializer='he_normal',
		            kernel_regularizer=regularizer, attn_kernel_regularizer=regularizer,
		            bias_regularizer=regularizer)([node_feats, adjacency_input])
		if residual:  # add a drug_residual_connection connection (as in the implementation of GATPredictor in DGL LifeSci)
			if concat_heads:
				res_feats = Dense(n_channels * n_attention_heads)(
					node_feats)  # according to the implementation of residual connections for GATConv in DGL
			else:
				res_feats = Dense(n_channels)(node_feats)
			x = add([x, res_feats])  # drug_residual_connection connection
		x = Dropout(dropout_rate)(x)
		node_feats = x
	x = GlobalSumPool()(x)

	submodel = Model(inputs=[nodes_input, adjacency_input], outputs=[x], name='drug_gat_submodel')

	return submodel


def dense_attention(inputs, feature_size, return_alphas=False):
	"""Dense attention layer"""
	# based on PaccMann code (dense_attention_layer): https://github.com/drugilsberg/paccmann/blob/master/paccmann/layers.py
	alphas = Dense(feature_size, activation='softmax', name='attention')(inputs)
	output = Multiply(name='filtered_with_attention')([inputs, alphas])

	if return_alphas:
		return (output, alphas)
	else:
		return output
