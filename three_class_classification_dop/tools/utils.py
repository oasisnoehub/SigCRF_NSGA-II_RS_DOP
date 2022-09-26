# %% raw
#### 使用Grid Research 进行模型选择
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve, precision_recall_fscore_support
from sklearn import metrics, svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize

"""
    功能：数据统计描述（）
    path: 数据文件地址
    label_name:标签变量
    is_plot：是否绘制热力图和类别统计图
    返回值：dataframe
"""
import time


def data_describe(path, label_name, is_plot=False):
    # 1. 导入数据

    df = pd.read_csv(path)
    print("=============data_describe================")
    print("检查数据是否存在null值：", df.isnull().values.any())
    print("检查数据存在null值的个数：", df.isnull().sum().sum())
    # drop nan值
    df.dropna(axis=0, how='any', inplace=True)
    print("处理完nan值之后检查数据是否存在null值：", df.isnull().values.any())
    print("label distribution : ")
    print(df[label_name].value_counts())
    print(df.describe())
    print("=============================")

    time_label = round(time.time())
    if (is_plot):
        # 绘制分析热力图+类统计图
        # 2. 热力图展示的是数值字段的相互相关性
        # sns.heatmap(df.corr(), annot=True)
        # 3. 绘制os 和 op 的 类统计信息
        # sns.set_theme(style="ticks", color_codes=True)
        # sns.catplot(data=df, x=label_name, kind="count")
        # 绘制所有变量相关图
        plt.figure(figsize=(18, 16))
        sns.pairplot(df, hue=label_name, palette='YlGnBu')
        plt.savefig('../images/variables_relative_' + str(time_label) + '.png', dpi=400)
        # plt.show()

    return df


def feature_uniform(df, cat_columns, num_columns, label_name):
    standardScaler = StandardScaler()
    num_features = standardScaler.fit_transform(df[num_columns])
    oneHotEncoder = OneHotEncoder(drop='first')  # 一般进行One-hot编码会加上drop=‘first’防止产生推导关系
    cat_features = oneHotEncoder.fit_transform(df[cat_columns]).toarray()  # 进行one-hot编码，并转换为array类型数据
    print("=============feature_uniform================")
    print("特征变量X + 标签向量y: ")
    X = np.hstack([cat_features, num_features])
    y = df[label_name].to_numpy()
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("X type:", type(X))
    print("y type:", type(y))
    print("=============================")
    return X, y


"""
    方法：过采样处理(导入imbalance-learn包)
"""


def imbalance_process(X, y, cat_columns, num_columns):
    print("=============imbalance_process================")
    # ==========================
    # 5. 进行过采样
    # 定义过采样器
    sampler = KMeansSMOTE(random_state=0)
    # 进行过采样
    X_res, y_res = sampler.fit_resample(X, y)
    # 将numpy.ndarray 转为 dataframe
    columns = cat_columns + num_columns
    features = pd.DataFrame(data=X_res, columns=columns)
    label = pd.DataFrame(data=y_res, columns=['OP_Group'])
    print("features shape: ", features.shape)
    print("label shape: ", label.shape)
    print("=============================")
    return features, label


# =================================
# rf_importance() ：使用随机森林进行重要性排序
# X_res：采样后特征举证
# y_res： 采样后标签向量
# cat_columns：分类变量名向量
# num_columns ： 数值变量名向量
# img_save_path ： 存特征重要性排序的图路径
# =================================

def rf_importance(X_train, X_test, y_train, y_test, cat_columns, num_columns, img_save_path):
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


# 绘制三分类混淆矩阵
def confusion_matrix_plot(X_test, y_test, classifier_trained, fold_index):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    class_names = ['nom', 'os', 'op']

    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier_trained,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
        save_path = "../images/confusion_matrix/" + title + '_' + str(fold_index + 1) + '.jpg'
        plt.savefig(save_path, dpi=600)
        # print(title)
        # print(disp.confusion_matrix)

    plt.show()


def multi_class_auroc(X_train, X_test, y_train, y_test, n_classes, classifier, is_plot=True):
    # print(type(X_train),type(y_train))
    # Learn to predict each class against the other
    y_train = label_binarize(y_train, classes=[1, 2, 3])
    y_test = label_binarize(y_test, classes=[1, 2, 3])
    # train model
    clf = classifier.fit(X_train, y_train)
    # predict
    y_pred = clf.predict(X_test)
    # score (macro)
    accuracy = clf.score(X_test, y_test)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    acu_result = []
    # Compute ROC curve and ROC area for each class
    if (is_plot):

        # ============plot AUROC curves==========
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            acu_result.append([i, roc_auc[i]])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        acu_result.append(["micro", roc_auc["micro"]])
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i + 1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.savefig("../images/cv_rf_fs.png",dpi=600)
        plt.show()

    return accuracy, precision, recall, fscore, acu_result


# =================================
# 定义函数 ：k_fold_cross_validation(cv,classifier)
# 功能：交叉验证
# 变量：
#     k_fold ： 交叉验证的折数
#     classifier ： 使用的分类器
#     X: 特征举证
#     y: 标签向量
# =================================
def kfold_cross_validation(X, y, k_fold, classifier, is_plot=False):
    cv_acc = []
    cv_precision = []
    cv_recall = []
    cv_fscore = []
    cv_auc = []
    cv_metrics = []
    # 定义交叉验证的折数
    cv = StratifiedKFold(n_splits=k_fold)

    for i, (train, test) in enumerate(cv.split(X, y)):
        accuracy, precision, recall, fscore, acu = multi_class_auroc(X_train=X[train],
                                                                     X_test=X[test],
                                                                     y_train=y[train],
                                                                     y_test=y[test],
                                                                     n_classes=3,
                                                                     classifier=classifier,
                                                                     is_plot=is_plot)
        # get different fold evaluation results
        cv_acc.append([accuracy])
        cv_precision.append([precision])
        cv_recall.append([recall])
        cv_fscore.append([fscore])
        cv_auc.append([acu])

    cv_metrics.append(['accuracy', np.mean(cv_acc), np.std(cv_acc)])
    cv_metrics.append(['precision', np.mean(cv_precision), np.std(cv_precision)])
    cv_metrics.append(['recall', np.mean(cv_recall), np.std(cv_recall)])
    cv_metrics.append(['fscore', np.mean(cv_fscore), np.std(cv_fscore)])
    cv_metrics.append(['acu', np.mean(cv_auc), np.std(cv_auc)])

    return cv_metrics
