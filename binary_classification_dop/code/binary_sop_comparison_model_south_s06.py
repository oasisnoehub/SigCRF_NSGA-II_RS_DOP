"""
    pre-condition : optimal feature set
    content : south optimal model compared with other models
"""
import sys

from binary_classification_dop.tools.opUtils import modelling_revised_calibration
from binary_classification_dop.tools.sop_tool import *


# ========================== file log ==========================

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


t = round(time.time())
log_name = "../logs/south_comparison_results_" + str(t) + ".log"
sys.stdout = Logger(log_name, sys.stdout)
# sys.stderr = Logger('results.log_file', sys.stderr)
# =====================================================================

label_name = "OP_Group"

final_feature_set = ['Retinopathy','Neuropathy',
                     'History of Smoking','Age','Weight','SBP',
                     'DBP','ALT','AST','HDL-C','LDL-C',
                     'Ca','P','FT4','PINP']

print("================================South Model =============================")

X_train_t_1 = pd.read_csv("../south_review_data/X_train_t.csv")
X_train_val_1 = pd.read_csv("../south_review_data/X_train_val.csv")
y_train_t_1 = pd.read_csv("../south_review_data/y_train_t.csv")
y_train_val_1 = pd.read_csv("../south_review_data/y_train_val.csv")

X_test_1 = pd.read_csv("../south_review_data/X_test.csv")
y_test_1 = pd.read_csv("../south_review_data/y_test.csv")

y_train_t_1 = np.array(y_train_t_1[label_name])
y_train_val_1 = np.array(y_train_val_1[label_name])
y_test_1 = np.array(y_test_1[label_name])


final_model = RandomForestClassifier(n_estimators=25,criterion='gini',random_state=42)

rf_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=final_model
                            )

print("===============================Logistic regression=============================")

lr_model = LogisticRegression(random_state=0)

selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=lr_model
                            )

print("===============================KNN=============================")

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=knn_model
                            )

print("===============================SVM =============================")

svm_model = SVC(C=15, gamma='auto', kernel='rbf', random_state=0,probability=True)

svmovr_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=svm_model
                            )

print("===============================DT=============================")

dt_model = DecisionTreeClassifier(random_state=0, max_depth=5)

dt_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=dt_model
                            )

print("==============================END=============================")