import os

from src.interpretability.interpret import ModelInterpreter
from src.dataset.dataset import MultiInputDataset
from src.models.drug_pairs_build_functions import expr_drug_dense_model


def test_deepshap():
	train_dataset = MultiInputDataset(response_dataset_path='test_files/split_data_train.csv.gz',
	                            id_cols=['NSC1', 'NSC2', 'CELLNAME'],
	                            output_col='COMBOSCORE',
	                            input_order=['expr', 'drugA', 'drugB'])
	train_dataset.load_expr(rnaseq_file='test_files/expr_scaled_train.npy')
	train_dataset.load_drugA(drugA_file='test_files/train_ECFP4_drugA.npy')
	train_dataset.load_drugB(drugB_file='test_files/train_ECFP4_drugB.npy')
	train_dataset.X_dict_to_list()
	dataset_dims = train_dataset.get_dataset_dimensions('expr_drug_dense_model')
	feature_names = {}
	for input_type in train_dataset.X_dict:
		features = [input_type + str(i) for i in range (train_dataset.X_dict[input_type].shape[1])]
		feature_names[input_type] = features
	train_dataset.feature_names = feature_names

	hyperparams = dataset_dims

	model = expr_drug_dense_model(**hyperparams)
	model.fit(train_dataset.X_dict, train_dataset.y, epochs=5, batch_size=64)
	#model.fit(train_dataset.X_list, train_dataset.y, epochs=5, batch_size=64)
	try:
		model_path = os.path.join('test_files', 'train_set_model')
		model.save(model_path)
	except Exception as e:
		print(e)
		model_path = os.path.join('test_files', 'train_set_model.h5')
		model.save(model_path)

	interpreter = ModelInterpreter(explainer_type='Deep', saved_model_path=model_path, dataset=train_dataset)
	shap_values = interpreter.compute_shap_values(train_dataset=train_dataset, n_background_samples=30)
	print(shap_values)

	interpreter.plot_feature_importance()
	interpreter.plot_feature_importance(plot_type='beeswarm')
	interpreter.plot_feature_importance(input_type='drugA', plot_type='beeswarm', max_display=50)
	interpreter.plot_feature_effect('drugA314') # difficult to interpret
	interpreter.plot_feature_effect('expr56') # also not easy to interpret
	interpreter.plot_sample_explanation(row_index=0)

	row_names = train_dataset.get_row_ids()
	interpreter.save_shap_values(row_ids=row_names, output_filepath='test_files/test_shap_deepexplainer.csv')
	interpreter.save_explanation('test_files/test_shap_deepexplainer_explanation_obj.pkl')


def test_gradientshap():
	train_dataset = MultiInputDataset(response_dataset_path='test_files/split_data_train.csv.gz',
	                                  id_cols=['NSC1', 'NSC2', 'CELLNAME'],
	                                  output_col='COMBOSCORE',
	                                  input_order=['expr', 'drugA', 'drugB'])
	train_dataset.load_expr(rnaseq_file='test_files/expr_scaled_train.npy')
	train_dataset.load_drugA(drugA_file='test_files/train_ECFP4_drugA.npy')
	train_dataset.load_drugB(drugB_file='test_files/train_ECFP4_drugB.npy')
	train_dataset.X_dict_to_list()
	dataset_dims = train_dataset.get_dataset_dimensions('expr_drug_dense_model')
	feature_names = {}
	for input_type in train_dataset.X_dict:
		features = [input_type + str(i) for i in range(train_dataset.X_dict[input_type].shape[1])]
		feature_names[input_type] = features
	train_dataset.feature_names = feature_names

	hyperparams = dataset_dims

	model = expr_drug_dense_model(**hyperparams)
	model.fit(train_dataset.X_dict, train_dataset.y, epochs=5, batch_size=64)
	# model.fit(train_dataset.X_list, train_dataset.y, epochs=5, batch_size=64)
	try:
		model_path = os.path.join('test_files', 'train_set_model')
		model.save(model_path)
	except Exception as e:
		print(e)
		model_path = os.path.join('test_files', 'train_set_model.h5')
		model.save(model_path)

	interpreter = ModelInterpreter(explainer_type='Gradient', saved_model_path=model_path, dataset=train_dataset)
	shap_values = interpreter.compute_shap_values(train_dataset=train_dataset, n_background_samples=30)
	print(shap_values)

	interpreter.plot_feature_importance()
	interpreter.plot_feature_importance(plot_type='beeswarm')
	interpreter.plot_feature_importance(input_type='drugA', plot_type='beeswarm', max_display=50)
	interpreter.plot_feature_effect('drugA314')
	interpreter.plot_feature_effect('expr56')
	interpreter.plot_sample_explanation(row_index=0)

	row_names = train_dataset.get_row_ids()
	interpreter.save_shap_values(row_ids=row_names, output_filepath='test_files/test_shap_gradientexplainer.csv')
	interpreter.save_explanation('test_files/test_shap_deepexplainer_explanation_obj.pkl')


if __name__ == '__main__':
	test_deepshap()
	test_gradientshap()