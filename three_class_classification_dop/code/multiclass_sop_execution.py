"""
 classification : three-class
 content : standard operation process(SOP) of SigRF-NSGA-II-RS
"""

from multiclass_sop_model import *
np.random.seed(42)

population_size = 32

num_generation = 50

num_features = 34

drop_rate = 0

mutation_rate = 0.2

model = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=42)

error_tolerance = 0
# =====================================
# multiclass standard operation process
multiclass_sop_model(population_size=population_size,
                     num_features=num_features,
                     num_generation=num_generation,
                     drop_rate=drop_rate,
                     mutation_rate=mutation_rate,
                     error_tolerance=error_tolerance,
                     model=model)