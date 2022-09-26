"""
    xgboost + shap explain models
"""
import numpy as np
import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt

np.random.seed(42)


num_columns = ['Age', 'Height', 'Weight', 'BMI',
               'SBP', 'DBP', 'Heart Rate', 'FBG',
               'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
               'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
               'Ca', 'P', 'FT3', 'FT4', 'VD3', 'N-MID', 'PINP', 'β-CTX']

cat_columns = ["Gender", "Macrovascular Complications",
               "History of Hypertension", "Nephropathy",
               "Retinopathy", "Neuropathy",
               "History of Smoking", "History of Drinking"]
label_name = "OP_Group"
# =====================================================================

X = pd.read_csv("../train_data/north_feature_data.csv")
y = pd.read_csv("../train_data/north_label_data.csv").to_numpy()

model = xgboost.XGBClassifier(random_state=0).fit(X, y)

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values,21)

# plt.savefig("../images/shap_south_top20.jpg",dpi=600)

