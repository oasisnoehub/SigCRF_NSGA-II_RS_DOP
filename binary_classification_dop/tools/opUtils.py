# 定义骨质疏松分析使用到的相应函数
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, ConfusionMatrixDisplay, \
    roc_auc_score
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold

# # 2. 热力图展示的是数值字段的相互相关性
# plt.figure(figsize=(20,18))
# sns.heatmap(df.corr(),annot=True)
#
# # 3. 绘制os 和 op 的 类统计信息
# sns.set_theme(style="ticks", color_codes=True)
# sns.catplot(data=df, x="OP_Group", kind="count")
# plt.show()
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize


# =================================
# plot_hotmap() ： 绘制变量相关性热力图
# =================================
def plot_hotmap(df):
    # 2. 热力图展示的是数值字段的相互相关性
    plt.figure(figsize=(20, 18))
    sns.heatmap(df.corr(), annot=True)
    plt.show()


# =================================
# plot_cat_distribute() ： 绘制类别变量统计图
# =================================
def plot_cat_distribute(df, label_name):
    # 3. 绘制os 和 op 的 类统计信息
    sns.set_theme(style="ticks", color_codes=True)
    sns.catplot(data=df, x=label_name, kind="count")
    plt.show()


# =================================
# df_basic_describe() ： 描述dataframe数据的基本信息
# path： 数据文件路径
# isPlot ： 是否绘制热力图和类别统计图
# num_columns ： 数值变量
# cat_columns： 分类变量
# label_name：标签变量名
# =================================
def df_basic_describe(path, isPlot, num_columns, cat_columns, label_name):
    # 1. 导入数据
    df = pd.read_csv(path)
    print("=============================")
    print("检查数据是否存在null值：", df.isnull().values.any())
    print("检查数据存在null值的个数：", df.isnull().sum().sum())
    # drop nan值
    df.dropna(axis=0, how='any', inplace=True)
    print("处理完nan值之后检查数据是否存在null值：", df.isnull().values.any())
    print("label distribution : ")
    print(df["OP_Group"].value_counts())
    print("=============================")
    # 绘制分析热力图+类统计图
    if (isPlot):
        # 2. 热力图展示的是数值字段的相互相关性
        plot_hotmap(df)
        # 3. 绘制os 和 op 的 类统计信息
        plot_cat_distribute(df, label_name)

    # 2.分类+数值变量处理
    standardScaler = StandardScaler()
    num_features = standardScaler.fit_transform(df[num_columns])
    oneHotEncoder = OneHotEncoder(drop='first')  # 一般进行One-hot编码会加上drop=‘first’防止产生推导关系
    cat_features = oneHotEncoder.fit_transform(df[cat_columns]).toarray()  # 进行one-hot编码，并转换为array类型数据
    print("特征变量X + 标签向量y: ")
    X = np.hstack([cat_features, num_features])
    y = df[label_name].to_numpy()
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("=============================")
    return X, y


# =================================
# unbalance_process() ： 对不平衡数据进行过采样
# X：特征矩阵
# y: 标签向量
# sampler_index： 选择要使用采样算法
#                 0:SMOTE
#                 1:BorderlineSMOTE
#                 2:KMeansSMOTE
#                 3:SVMSMOTE
# =================================

def unbalance_process(X, y, sampler_index):
    # 定义各种sampler
    samplers = [
        SMOTE(random_state=0),
        BorderlineSMOTE(random_state=0, kind="borderline-1"),
        KMeansSMOTE(random_state=0),
        SVMSMOTE(random_state=0),
    ]
    # 选择一个sampler改善不平衡数据
    X_res, y_res = samplers[sampler_index].fit_resample(X, y)
    print("=============================")
    print("X 采样后的shape：", X_res.shape, "X 采样后的type：", type(X_res))
    print("y 采样后的shape：", y_res.shape, "y 采样后的type：", type(y_res))
    print("y 采样后的value_counts：")
    print(pd.value_counts(y_res))
    print("=============================")
    return X_res, y_res


# =================================
# unbalance_process() ： 对不平衡数据进行过采样
# X_res：特征矩阵（过采样后）
# y_res: 标签向量（采样后）
# model： 需要进行训练的模型
# =================================
def modelling_evaluate(X_res, y_res, model):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 测试模型
    prediction = model.predict(X_test)
    prediction_proba = model.predict_proba(X_test)[:, 1]

    # 评估模型
    accuracy = metrics.accuracy_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    recall = metrics.recall_score(y_test, prediction)
    AUC = metrics.roc_auc_score(y_test, prediction_proba)
    print("=============================")
    print('track --- model : ', type(model))
    print('track --- accuracy : ', accuracy)
    print('track --- precision : ', precision)
    print('track --- recall : ', recall)
    print('track --- AUC : ', AUC)
    print("=============================")


# =================================
# rf_importance() ：使用随机森林进行重要性排序
# X_res：采样后特征举证
# y_res： 采样后标签向量
# cat_columns：分类变量名向量
# num_columns ： 数值变量名向量
# img_save_path ： 存特征重要性排序的图路径
# =================================
def rf_importance(X_res, y_res, cat_columns, num_columns, img_save_path):
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    # print(importances)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    feature_labels = cat_columns + num_columns

    forest_importances = pd.Series(importances, index=feature_labels)

    # 对重要性进行降序排列 并 输出
    sort_importance = forest_importances.sort_values(ascending=False)
    print("=============================")
    print("ALL Feature Importance: ")
    print(sort_importance)

    fig, ax = plt.subplots(figsize=(10, 10))
    sort_importance.plot.barh(ax=ax)

    # ax.set_yticklabels(feature_labels)
    plt.title("Feature Importances", fontsize=18)
    plt.xlabel("Importance", fontsize=18)
    plt.xlim(0, 0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.savefig(img_save_path, dpi=600)
    plt.show()
    print("=============================")
    # 打印重要程度前15个特征
    print("top 15 features:")
    for i in range(len(sort_importance)):
        if i == 15:
            print(sort_importance[:i])
    print("=============================")


# =================================
# osop_process() ：进行整个预处理建模和评估
# path_switch：1(南方数据) else（北方数据）
# model： 需要进行训练的模型
# isPlot：选择是否绘制分析信息图
# =================================
def all_osop_process(model, isPlot):
    # 进行预测

    path = "../readydata/all_osop_data.csv"

    num_columns = ['Age', 'Height', 'Weight', 'BMI',
                   'SBP', 'DBP', 'Heart Rate', 'FBG',
                   'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
                   'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
                   'Ca', 'P', 'FT3', 'FT4', 'VD3', 'N-MID', 'PINP', 'β-CTX']

    cat_columns = ["Sex", "Macrovascular Complications",
                   "History of Hypertension", "Nephropathy",
                   "Retinopathy", "Neuropathy",
                   "History of Smoking", "History of Drinking"]
    label_name = "OP_Group"

    # 1.描述一下df的基本信息情况

    X, y = df_basic_describe(
        path=path,
        isPlot=isPlot,
        num_columns=num_columns,
        cat_columns=cat_columns,
        label_name=label_name
    )

    # 2. 使用采样方法：SMOTE 对不平衡数据进行过采样
    sampler_index = 0
    X_res, y_res = unbalance_process(
        X=X,
        y=y,
        sampler_index=sampler_index
    )
    # 特征重要性分析
    img_save_path = "../images/feature importance RF using permutation.png"
    rf_importance(
        X_res=X_res,
        y_res=y_res,
        cat_columns=cat_columns,
        num_columns=num_columns,
        img_save_path=img_save_path
    )

    # 训练模型
    modelling_evaluate(X_res, y_res, model)
    classifier_roc_analysis(X_res, y_res, model, 't1')


# =================================
# osop_process() ：进行整个预处理建模和评估
# path_switch：1(南方数据) else（北方数据）
# model： 需要进行训练的模型
# isPlot：选择是否绘制分析信息图
# =================================
def osop_process(path_switch, model, isPlot):
    # 进行预测
    path_north = "../readydata/north_osop_data_label_unit.csv"
    path_south = "../readydata/south_osop_data_label_unit.csv"

    num_columns = ['Age', 'Height', 'Weight', 'BMI',
                   'SBP', 'DBP', 'Heart Rate', 'FBG',
                   'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
                   'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
                   'Ca', 'P', 'FT3', 'FT4', 'VD3', 'N-MID', 'PINP', 'β-CTX']

    cat_columns = ["Sex", "Macrovascular Complications",
                   "History of Hypertension", "Nephropathy",
                   "Retinopathy", "Neuropathy",
                   "History of Smoking", "History of Drinking"]
    label_name = "OP_Group"

    # 1.描述一下df的基本信息情况
    # 选择要处理的数据路径
    if (path_switch == 1):
        path = path_south
    else:
        path = path_north

    X, y = df_basic_describe(
        path=path,
        isPlot=isPlot,
        num_columns=num_columns,
        cat_columns=cat_columns,
        label_name=label_name
    )

    # 2. 使用采样方法：KMeansSMOTE 对不平衡数据进行过采样
    sampler_index = 0
    if (path_switch == 1):
        X_res = X
        y_res = y
    else:
        X_res, y_res = unbalance_process(
            X=X,
            y=y,
            sampler_index=sampler_index
        )
    # 特征重要性分析
    img_save_path = "../images/feature importance RF using permutation.png"
    rf_importance(
        X_res=X_res,
        y_res=y_res,
        cat_columns=cat_columns,
        num_columns=num_columns,
        img_save_path=img_save_path
    )

    # 训练模型
    modelling_evaluate(X_res, y_res, model)
    classifier_roc_analysis(X_res, y_res, model, 't1')


# =================================
# 定义函数 ：classfier_roc_analysis
# 功能：交叉验证 + 训练分类器 + 绘制ROC曲线
# 变量：
#     cv ： 交叉验证的折数
#     classifier ： 使用的分类器
# =================================
def classifier_roc_analysis(X, y, classifier, name=None):
    tprs = []
    aucs = []

    clf_accuracy = []
    clf_precision = []
    clf_recall = []
    clf_f1_score = []

    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ### 定义交叉验证的折数
    cv = StratifiedKFold(n_splits=10)

    for i, (train, test) in enumerate(cv.split(X, y)):
        model = classifier.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        # other metrics
        clf_accuracy.append(metrics.accuracy_score(y[test], y_pred))
        clf_precision.append(metrics.precision_score(y[test], y_pred))
        clf_recall.append(metrics.recall_score(y[test], y_pred))
        clf_f1_score.append(metrics.f1_score(y[test], y_pred))

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC"
    )
    ax.legend(loc="lower right")
    plt.title(name + ' ROC', fontsize=18)
    plt.xlabel('FPR', fontsize=18)
    plt.ylabel('TPR', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if name:
        plt.savefig('../images/' + name + '.jpg', dpi=600)
    plt.show()

    # 将预测结果返回（means ± standard）
    results_means = []
    results_std = []
    # mean
    results_means.append(np.mean(clf_accuracy))
    results_means.append(np.mean(clf_precision))
    results_means.append(np.mean(clf_recall))
    results_means.append(np.mean(clf_f1_score))
    results_means.append(np.mean(aucs))
    # std
    results_std.append(np.std(clf_accuracy))
    results_std.append(np.std(clf_precision))
    results_std.append(np.std(clf_recall))
    results_std.append(np.std(clf_f1_score))
    results_std.append(np.std(aucs))

    return results_means, results_std


# ========================
# 保留4位小数
def round_num_4(num):
    return round(num, 4)


# ========================
# 获取模型准确度
def acc_score(X_train, X_test, y_train, y_test, classifiers, models):
    score = pd.DataFrame({"Classifier": classifiers})
    index = 0
    acc = []
    for m in models:
        model = m
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, predictions))
        index += 1
    score["Accuracy"] = acc
    score.sort_values(by="Accuracy", ascending=False, inplace=True)
    score.reset_index(drop=True, inplace=True)
    return score


# =================================

# =================================
# modelling_revised() ： 对nsga-ii和rf的到的特征方案模型进行复现
# X_train ： 最终诊断方案的特征矩阵训练数据
# y_train ： 最终诊断方案的标签向量训练数据
# X_test ：最终诊断方案的特征矩阵测试数据
# y_test ：最终诊断方案的标签向量测试数据
# model ： 模型
# =================================
def modelling_revised(X_train, y_train, X_test, y_test, model):
    # 训练模型
    clf = model.fit(X_train, y_train)
    # 测试模型
    y_pred = clf.predict(X_test)
    # prediction_proba = clf.predict_proba(X_test)[:, 1]
    mean_acc = clf.score(X_test, y_test)

    # confusion matricx
    conf_ma = confusion_matrix(y_test, y_pred)
    conf_ma_scaled = confusion_matrix(y_test, y_pred, normalize='true')
    #
    # 评估模型
    print("=============================")
    selected_columns = X_train.columns.values
    print(selected_columns)
    print('track --- feature length : ', len(selected_columns))
    print('track --- model : ', type(model))
    print('track --- mean_acc : ', round_num_4(mean_acc))
    print('track ---1- mean_acc : ', round_num_4(1 - mean_acc))
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print('track --- precision: ', round_num_4(precision))
    print('track --- recall: ', round_num_4(recall))
    print('track --- fscore: ', round_num_4(fscore))
    print('track --- conf_ma : ')
    print(conf_ma)
    print('track --- conf_ma_scaled : ')
    print(conf_ma_scaled)
    print("=============================")

    return selected_columns


# 矫正分类器模型重建
def modelling_revised_calibration(X_train, X_train_val, y_train, y_train_val, X_test, y_test, model):
    # 训练模型
    clf = model.fit(X_train, y_train)
    # 进行classifier的矫正
    cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    cal_clf.fit(X_train_val, y_train_val)
    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = np.transpose(clf.predict_proba(X_test)[:,1])
    y_cal_pred = cal_clf.predict(X_test)
    y_cal_pred_proba = np.transpose(cal_clf.predict_proba(X_test)[:,1])
    # 评估
    # uncalibrated model
    mean_acc = clf.score(X_test, y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    fscore = metrics.f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test,y_pred_proba)

    # calibrated model
    cal_acc = metrics.accuracy_score(y_test, y_cal_pred)
    cal_precision = metrics.precision_score(y_test, y_cal_pred)
    cal_recall = metrics.recall_score(y_test, y_cal_pred)
    cal_fscore = metrics.f1_score(y_test, y_cal_pred)
    cal_auroc =  roc_auc_score(y_test,y_cal_pred_proba)


    # confusion matricx
    conf_ma = confusion_matrix(y_test, y_pred)
    conf_ma_scaled = confusion_matrix(y_test, y_pred, normalize='true')
    cal_conf_ma = confusion_matrix(y_test, y_cal_pred)
    cal_conf_ma_scaled = confusion_matrix(y_test, y_cal_pred, normalize='true')
    # 评估模型
    print("=============================")
    selected_columns = X_train.columns.values
    print(selected_columns)
    print('track --- feature length : ', len(selected_columns))
    # 模型
    print('track --- model : ', type(model))
    print('track --- calibration model : ', type(cal_clf))
    # 评价指标
    print("================================")
    print('track --- mean_acc : ', mean_acc)
    print('track --- acc : ', round_num_4(acc), ' calibration : ', round_num_4(cal_acc))
    print('track ---1- mean_acc : ', round_num_4(1 - acc), ' calibration : ', round_num_4(1 - cal_acc))
    print('track --- precision: ', round_num_4(precision), ' calibration : ', round_num_4(cal_precision))
    print('track --- recall: ', round_num_4(recall), ' calibration : ', round_num_4(cal_recall))
    print('track --- fscore: ', round_num_4(fscore), ' calibration : ', round_num_4(cal_fscore))
    print('track --- 1- fscore: ', round_num_4(1-fscore), ' calibration : ', round_num_4(1-cal_fscore))
    print("==============original================")
    print('track --- auroc: ')
    print(auroc)
    print('track --- conf_ma : ')
    print(conf_ma)
    print('track --- conf_ma_scaled : ')
    print(conf_ma_scaled)
    print("==============after calibration================")
    print('track --- auroc : ')
    print(cal_auroc)
    print('track --- conf_ma : ')
    print(cal_conf_ma)
    print('track --- conf_ma_scaled : ')
    print(cal_conf_ma_scaled)
    print("======================================")


# 绘制二分类混淆矩阵
def binary_confusion_matrix_plot(y_test, y_pred, name=None):
    plt.figure(figsize=(8, 7))
    plt.tick_params(labelsize=15)
    font = {'size': 12}
    plt.rc('font', **font)  # pass in the font dict as kwargs
    np.set_printoptions(precision=2)
    t = round(time.time())
    # Plot non-normalized confusion matrix

    class_names = ['Normal', 'Osteoporosis']

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize="true",
    )
    title = "Normalized Confusion Matrix"
    disp.ax_.set_title(title)
    plt.gcf().subplots_adjust(left=0.22, top=0.91, bottom=0.09)  # 在此添加修改参数的代码
    save_path = "../images/" + title + "(" + name + ")" + str(t) + '.jpg'
    plt.savefig(save_path, dpi=600)

