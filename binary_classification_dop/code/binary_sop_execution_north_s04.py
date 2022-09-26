"""
 dataset: north dataset
 content : standard operation process(SOP) of SigRF-NSGA-II-RS

"""

from binary_sop_model_north_s03 import *

np.random.seed(42)
# 定义种群数
population_size = 32
# 迭代次数
num_generation = 50
# 定义总特征数
num_features = 34
# 初始化丢掉的特征数占比 drop_rate
drop_rate = 0
# 变异比例
mutation_rate = 0.2
# 基础分类器模型
# 定义基础分类器
model = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=42)
# 1.设置牺牲容忍度
error_tolerance = 0
# =====================================
print("========binary classification SOP : North data========")
binary_sop_model_north(population_size=population_size,
                     num_features=num_features,
                     num_generation=num_generation,
                     drop_rate=drop_rate,
                     mutation_rate=mutation_rate,
                     error_tolerance=error_tolerance,
                     model=model)