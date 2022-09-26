
from binary_classification_dop.tools.utils import *
np.random.seed(0)

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


# 数据信息描述
df = data_describe(path,label_name,is_plot=True)

# # 分类变量与数值变量归一化处理
X,y = feature_uniform(df=df,
                      cat_columns=cat_columns,
                      num_columns=num_columns,
                      label_name=label_name)

