import pickle
import pandas as pd

with open('../results/2021-10-26_14-36-02/train_model.pkl', 'rb') as f:
	lgbm_model = pickle.load(f)

with open('../results/2021-09-27_23-01-20/train_model.pkl', 'rb') as f:
	rf_model = pickle.load(f)

with open('../data/nci_almanac_preprocessed/targets_ecfp4_model_feature_names.pkl', 'rb') as f:
	feature_names_dict = pickle.load(f)
feature_names = feature_names_dict['drugA'] + feature_names_dict['drugB'] + feature_names_dict['expr']

lgbm_feature_importance = lgbm_model.feature_importances_
rf_feature_importance = rf_model.feature_importances_

rf_df = pd.DataFrame(data={'Feature': feature_names, 'Importance': rf_feature_importance})
rf_df.sort_values(by='Importance', ascending=False, inplace=True)
rf_df.to_csv('results/rf_feature_importance.csv', index=False)

lgbm_df = pd.DataFrame(data={'Feature': feature_names, 'Importance': lgbm_feature_importance})
lgbm_df.sort_values(by='Importance', ascending=False, inplace=True)
lgbm_df.to_csv('results/lgbm_feature_importance.csv', index=False)