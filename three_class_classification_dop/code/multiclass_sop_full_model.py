"""
    full model with 34 features
"""

from three_class_classification_dop.tools.opUtils import modelling_revised_calibration
from three_class_classification_dop.tools.sop_tool import *

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


print("================================Full Model review=============================")

X_train_t_1 = pd.read_csv("../review_data/X_train_t.csv")
X_train_val_1 = pd.read_csv("../review_data/X_train_val.csv")
y_train_t_1 = pd.read_csv("../review_data/y_train_t.csv")
y_train_val_1 = pd.read_csv("../review_data/y_train_val.csv")

X_test_1 = pd.read_csv("../review_data/X_test.csv")
y_test_1 = pd.read_csv("../review_data/y_test.csv")

y_train_t_1 = np.array(y_train_t_1[label_name])
y_train_val_1 = np.array(y_train_val_1[label_name])
y_test_1 = np.array(y_test_1[label_name])

# ==========================


final_model = RandomForestClassifier(n_estimators=25,criterion='gini',random_state=42)

selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1,
                                X_train_val=X_train_val_1,
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1,
                                y_test=y_test_1,
                                model=final_model
                            )


