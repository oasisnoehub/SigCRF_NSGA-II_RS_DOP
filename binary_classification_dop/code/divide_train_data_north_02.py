"""
    North preprocessing
"""

from binary_classification_dop.tools.utils import *
from binary_classification_dop.tools.sop_tool import *

np.random.seed(42)

# ========================== log file ==========================
import sys
import time
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
log_name = "../logs/results_"+str(t)+".log"
sys.stdout = Logger(log_name, sys.stdout)
# sys.stderr = Logger('results.log_file', sys.stderr)
# =====================================================================

path = "../data/north_nomop_data.csv"
isPlot = True

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

columns = np.concatenate((cat_columns,num_columns))

df = data_describe(path,label_name)

X,y = feature_uniform(df=df,
                      cat_columns=cat_columns,
                      num_columns=num_columns,
                      label_name=label_name)

feature_data,label_data = imbalance_process(X=X,
                                            y=y,
                                            cat_columns=cat_columns,
                                            num_columns=num_columns)

feature_data_path = "../train_data/north_feature_data.csv"
feature_data.to_csv(feature_data_path, sep=',', index=False)
label_data_path = "../train_data/north_label_data.csv"
label_data.to_csv(label_data_path, sep=',', index=False)
