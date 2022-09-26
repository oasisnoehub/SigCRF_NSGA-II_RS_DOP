
# encoding : utf-8
# 使用 NSGA-II 算法

"""
使用NSGA-II 和 随机森林 + 定制的评分系统 RS =》 筛选出
"""

import pandas as pd
import numpy as np
import imageio
import time

from imblearn.over_sampling import KMeansSMOTE
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tools import nsga_tool
from tools.opUtils import modelling_revised

np.random.seed(42)

"""
    描述：dop_sop（）按照NSGA-II-RF+RS评分系统流程的建模标准处理流程 SOP
    path : 数据集文件路径
    num_generation ：遗传迭代数
    model_index : 选择基础模型参数
        0：svm.SVC(kernel='linear'）
        1：svm.SVC(kernel='rbf',random_state=42)
        2：svm.SVC(kernel='sigmoid',random_state=42,C=1)
        3：LogisticRegression(max_iter = 100,random_state=42)
        4：RandomForestClassifier(n_estimators=200, random_state=42)
        5：AdaBoostClassifier(random_state = 42)
        6：DecisionTreeClassifier(random_state=42)
        7：KNeighborsClassifier()
        8: GradientBoostingClassifier(random_state=42)
    debug_2：是否将得到的每代目标值空间图转换为gif图
"""
def dop_sop(path,num_generation=5,debug_2=False):


    """
    （1）NSGA-II-RF算法流程
    """
    # ==========================
    # 1. 读取数据
    df = pd.read_csv(path)
    # ==========================
    # 2. 处理分类变量（one-hot 编码）
    cat_columns = ["Sex","Macrovascular Complications","History of Hypertension","Nephropathy","Retinopathy","Neuropathy","History of Smoking","History of Drinking"]
    oneHotEncoder = OneHotEncoder(drop='first') # 一般进行One-hot编码会加上drop=‘first’防止产生推导关系
    cat_features = oneHotEncoder.fit_transform(df[cat_columns]).toarray() # 进行one-hot编码，并转换为array类型数据
    # ==========================
    # 3. 处理数值变量（标准化）
    num_columns = ['Age','Height', 'Weight', 'BMI',
           'SBP', 'DBP', 'Heart Rate', 'FBG',
           'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
           'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
           'Ca', 'P', 'FT3', 'FT4','VD3', 'N-MID', 'PINP', 'β-CTX']
    standardScaler = StandardScaler()
    num_features = standardScaler.fit_transform(df[num_columns])
    # ==========================
    # 4. 获取特征矩阵 X 和 标签向量 y
    columns = np.concatenate((cat_columns,num_columns))
    X = np.hstack([cat_features, num_features])
    y = df["OP_Group"].to_numpy()
    # ==========================
    # 5. 进行过采样
    # 定义过采样器
    sampler = KMeansSMOTE(random_state=0)
    # 进行过采样
    X_res, y_res = sampler.fit_resample(X, y)
    # 将numpy.ndarray 转为 dataframe
    features = pd.DataFrame(data=X_res,columns=columns)
    labels = pd.DataFrame(data=y_res,columns=['OP_Group'])
    # ==========================
    # 6.划分数据集(80:20)
    X_train, X_test, y_train, y_test = nsga_tool.split(features, labels)
    # ==========================
    """
    选择基础分类器（此处选择的随机森林）+ 参数设置
    """
    # ==========================
    # 定义种群数
    population_size = 32
    # 迭代次数
    # num_generation = 5
    # 定义总特征数
    num_features = 34
    # 初始化丢掉的特征数占比 drop_rate
    drop_rate = 0
    # 变异比例
    mutation_rate = 0.2
    # 基础分类器模型
    # 定义基础分类器
    base_model = SVC(C=15, gamma='auto', kernel='rbf', random_state=0)
    model = OneVsRestClassifier(base_model)
    # ==========================
    # 7.执行程序（NSGA-II-RF）
    features, top_object,front_0_num = nsga_tool.generations_v2(
                                                    X_train=X_train,
                                                    y_train=y_train,
                                                    X_test=X_test,
                                                    y_test=y_test,
                                                    model= model,
                                                    pop_size=population_size,
                                                    num_features = num_features,
                                                    drop_rate= drop_rate,
                                                    mutation_rate = mutation_rate,
                                                    num_generation = num_generation
                                                )
    # ==========================

    """
    （2）自定义评分流程
    """
    # ==========================
    # 1.设置牺牲容忍度
    error_tolerence = 0.005

    # 2.自定义评分（从帕累托解集中选出最佳模型）
    best_model_selected = nsga_tool.sort_selected_final_model(
                                        features=features,
                                        columns=columns,
                                        top_object=top_object,
                                        front_0_num=front_0_num,
                                        error_toler=error_tolerence
                                        )

    # ==========================
    # 3.得到最佳模型的目标值以及选择的特征变量
    final_model_features = []
    final_model_target_val = []
    print('front_0_num: ', front_0_num)
    for index in range(front_0_num):
            if(top_object[index] == best_model_selected):
                print("==============⚠️ NOTICE! THIS IS THE BEST MODEL!==============")
                final_model_features = columns[features[index]]
                final_model_target_val = top_object[index]
                print('track ---best_model_selected: ', best_model_selected)
            print(f"track ========= {index} =========")
            print(f'track ---{index}-- columns[features[index]] len: ',len(columns[features[index]]))
            print(f'track ---{index}-- columns[features[index]] : ',columns[features[index]])
            print(f'track ---{index}-- top_object: ', top_object[index])


    # ==========================

    """
    （3）对产出的最佳模型方案进行建模输出复现
    """
    # ==========================
    # 1.重新实例化模型（与之前使用的基础分类器类型一致）
    base_model = SVC(C=15, gamma='auto', kernel='rbf', random_state=0)
    final_model = OneVsRestClassifier(base_model)

    X_train_selected = X_train[final_model_features]
    X_test_selected = X_test[final_model_features]
    # 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
    modelling_revised(
        X_train=X_train_selected,
        y_train=y_train,
        X_test=X_test_selected,
        y_test=y_test,
        model=final_model
    )
    # ==========================


    """
    （4）* 将每个generation得到的目标绘制为gif图 (optional)
    """
    # ==========================
    # 将每个generation得到的目标绘制为gif图
    if debug_2 :
        # 获取时间戳
        now = int(round(time.time()*1000))
        now02 = time.strftime('%Y%m%d-%H%M%S',time.localtime(now/1000))

        gif_images = []
        gif_n_generation = num_generation
        for i in range(1,gif_n_generation+1):
            gif_images.append(imageio.imread(f'../gaImages/generation_{i}.png'))   # 读取图片
        imageio.mimsave(f"../gifs/{now02}-nsga-ii.gif", gif_images, fps=3)   # 转化为gif动画
    # ==========================

    return X_train,X_train_selected,y_train,X_test,X_test_selected,y_test,final_model_features,final_model_target_val


def comparison_models(X_train,y_train,X_test,y_test,model,name="None"):
    # ==================
    # 计算CPU训练时间
    start = time.time()
    clf = model.fit(X_train, y_train)
    end = time.time()
    train_time = round_num_4((end - start)*1000)
    print('track -- model type:',type(model))
    print('training time: %s Seconds' , train_time)
    # ==================
    # 预测评估模型
    prediction = clf.predict(X_test)
    prediction_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = round_num_4(metrics.accuracy_score(y_test, prediction))
    precision = round_num_4(metrics.precision_score(y_test, prediction))
    recall = round_num_4(metrics.recall_score(y_test, prediction))
    F1_score = round_num_4(metrics.f1_score(y_test, prediction))
    AUC = round_num_4(metrics.roc_auc_score(y_test, prediction_proba))
    metrics_aprfa = [accuracy,precision,recall,F1_score,AUC]
    # ==================
    # 绘制混淆矩阵
    confusion_matrix_plot(X_test, y_test, clf,name)
    # ==================
    # 绘制特征重要性排序
    #
    feature_columns = X_train.columns.values
    img_save_path = "../images/importances/rf_importance.jpg"
    rf_importance_rank(clf, feature_columns,img_save_path)

    return metrics_aprfa

# =======================
# 绘制随机森林特征重要性
def rf_importance_rank(forest,feature_columns,img_save_path):

    importances = forest.feature_importances_
    # print(importances)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    feature_labels = feature_columns

    forest_importances = pd.Series(importances, index=feature_labels)

    # 对重要性进行降序排列 并 输出
    sort_importance = forest_importances.sort_values(ascending=False)
    print("=============================")
    print("ALL Feature Importance: ")
    print(sort_importance)

    fig, ax = plt.subplots(figsize=(10, 10))
    sort_importance.plot.barh(ax=ax)

    # ax.set_yticklabels(feature_labels)
    plt.title("特征重要性排序", fontsize=23)
    plt.xlabel("重要性", fontsize=23)
    plt.xlim(0, 0.4)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.savefig(img_save_path, dpi=600)
    plt.show()
    # print("=============================")
    # # 打印重要程度前15个特征
    # print("top 15 features:")
    # for i in range(len(sort_importance)):
    #     if i == 15:
    #         print(sort_importance[:i])
    # print("=============================")


# ========================
# 保留4位小数
def round_num_4(num):
    return round(num,4)
# ========================
