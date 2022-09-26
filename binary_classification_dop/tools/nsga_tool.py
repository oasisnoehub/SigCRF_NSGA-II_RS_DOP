# encoding: utf-8
# GA feature selection
import math
import random
import time
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from random import randint

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import  train_test_split
from sklearn import metrics
import warnings

from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

np.random.seed(42)
# ==========================
# 解决中文乱码
plt.rcParams['font.sans-serif']= ['Heiti TC']#防止中文乱码
plt.rcParams['axes.unicode_minus']=False#解决负号'-'显示为方块的问题

# 划分训练数据集和测试数据集(80:20)
def split(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

'''
函数：获取筛选特征后的训练集和测试数据集
'''
def train_and_test_model(features_data,select_features,labels,model,model_nam):

    results_dict = {}
    results_list = []

    selected_feature_data = features_data[select_features]

    X_train, X_test, y_train, y_test = split(selected_feature_data, labels)
    # ==================
    # 计算CPU训练时间
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time =  (end - start)
    print(f'{model_nam} training time: %s Seconds' %(train_time))
    # ==================
    # 预测
    prediction = model.predict(X_test)
    prediction_proba = model.predict_proba(X_test)[:,1]
    # 预测准确度
    accuracy = metrics.accuracy_score(y_test,prediction)
    precision = metrics.precision_score(y_test,prediction)
    recall = metrics.recall_score(y_test,prediction)
    F1_score = metrics.f1_score(y_test,prediction)
    AUC = metrics.roc_auc_score(y_test,prediction_proba)

    results_dict['accuracy'] = round(float(accuracy),4)
    results_dict['precision'] = round(float(precision),4)
    results_dict['recall'] = round(float(recall),4)
    results_dict['F1_score'] = round(float(F1_score),4)
    results_dict['AUC'] = round(float(AUC),4)

    results_list.append(round(float(accuracy),4))
    results_list.append(round(float(precision),4))
    results_list.append(round(float(recall),4))
    results_list.append(round(float(F1_score),4))
    results_list.append(round(float(AUC),4))

    # print('1-accuracy: %.4f' %float(1-accuracy))

    return results_dict,results_list, train_time

# 获取模型准确度
def acc_score(X_train,X_test,y_train,y_test,classifiers,models):
    score = pd.DataFrame({"Classifier":classifiers})
    index = 0
    acc = []
    for m in models:
        model = m
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        acc.append(metrics.accuracy_score(y_test,predictions))
        index += 1
    score["Accuracy"] = acc
    score.sort_values(by="Accuracy",ascending=False,inplace=True)
    score.reset_index(drop=True,inplace=True)
    return score

# ==== genetic algorithm for feature selection 遗传算法====
# 1. 初始化种群
def initilization_population(pop_size,num_feature,drop_rate=0.3):
    population = []
    for i in range(pop_size):
        chromosome = np.ones(num_feature,dtype=np.bool) # 编码（bool类型）
        chromosome[:int(drop_rate*num_feature)] = False # 0.3*num_feature 个特征为false = 不选
        np.random.shuffle(chromosome) # shuffle 染色体编码
        population.append(chromosome) # 将染色体添加到种群中
    return population

# 2. 适应度计算（评估分数 = 模型预测的准确度）
from sklearn.utils.multiclass import type_of_target
def fit_score(population,model,X_train,X_train_val,X_test,y_train,y_train_val,y_test):
    # 划分数据集（8：2）
    scores = []
    # 对种群中所有的染色体计算适应度（一个染色体代表一个特征组合）
    for chromosome in population:
        # 以当前染色体的特征组合训练模型
        clf = model.fit(X_train.iloc[:, chromosome],y_train)
        # 进行classifier的校准训练+5fold CV
        cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
        cal_clf.fit(X_train_val.iloc[:, chromosome], y_train_val)
        # 预测
        y_pred = clf.predict(X_test.iloc[:,chromosome])
        # y_pred_proba = clf.predict_proba(X_test.iloc[:, chromosome])

        y_cal_pred = cal_clf.predict(X_test.iloc[:,chromosome])
        # y_cal_pred_proba = cal_clf.predict_proba(X_test.iloc[:, chromosome])
        # 评价分数
        fscore = metrics.f1_score(y_test, y_pred)
        # cal_acc = metrics.accuracy_score(y_test, y_cal_pred)
        # cal_precision = metrics.precision_score(y_test, y_cal_pred)
        # cal_recall = metrics.recall_score(y_test, y_cal_pred)
        cal_fscore = metrics.f1_score(y_test, y_cal_pred)
        # cal_accuracy = metrics.accuracy_score(y_test,y_cal_pred)
        # cal_precision, cal_recall, cal_fscore, _ = precision_recall_fscore_support(y_test, y_cal_pred,average='binary')
        # 添加目标分数
        scores.append(fscore)

    # 将分数和种群转为array
    scores = np.array(scores)
    population = np.array(population)
    inds = np.argsort(scores) # 按分数进行排序(从最高到最低降序排列，同时保证分数和染色体对应一致)
    # front_1 = non_dominated_sort(scores,population)
    desc_score = list(scores[inds][::-1])
    desc_population = list(population[inds,:][::-1])
    return desc_score,desc_population

# 获取population的targets（target_1 + target_2）
def get_targets(scores,population):
    # 获取 scores对应染色体 选取到的特征数
    chroms_obj_record = []
    for i in range(len(scores)):
        # 获取当前score 对应选取的特征数
        target_1 = 0
        target_2 = 0
        targets = []
        chromosome = population[i]
        num_of_features = 0
        for c in chromosome:
            if c:
                num_of_features = num_of_features+1
        # 目标1 ： min (1 - accuracy)
        target_1 = 1 - scores[i]
        # 目标2 ： min (number of features)
        target_2 = num_of_features
        # 将每个染色体得到的目标值合并
        targets.append(target_1)
        targets.append(target_2)
        chroms_obj_record.append(targets)
    # test
    # print('track--get_target population[0]: ',population[0])
    # print('track--get_target scores[0]: ',scores[0])
    # print('track--get_target chroms_obj_record[0]: ',chroms_obj_record[0])
    return population,chroms_obj_record

# 非支配排序 non-dominated sorting (ranking)
# population : 产生的子群
# scores：子群中每个染色体得到的分数
# chroms_obj_record : 记录的每个chromo的目标值--
# chroms_obj_record[index][0] = 当前index染色体的1-accuracy
# chroms_obj_record[index][1] = 当前index染色体的特征数
def non_dominated_sorting(population_size, chroms_obj_record):
    s, n = {}, {}
    front, rank = {}, {}
    front[0] = []
    for p in range(population_size):
        s[p] = [] # 支配了谁
        n[p] = 0 # 被几个人支配
        for q in range(population_size):

            if ((chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] < chroms_obj_record[q][
                1]) or (chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <
                        chroms_obj_record[q][1])
                    or (chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] <=
                        chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] > chroms_obj_record[q][
                1]) or (chroms_obj_record[p][0] >= chroms_obj_record[q][0] and chroms_obj_record[p][1] >
                        chroms_obj_record[q][1])
                  or (chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] >=
                      chroms_obj_record[q][1])):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front[i] = Q

    del front[len(front) - 1] # 删除最后一个front
    return front
# 计算拥挤度
def calculate_crowding_distance(front, chroms_obj_record):
    distance = {m: 0 for m in front}
    for o in range(2):
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        # 数值 999999999999 代表边界点的距离无穷大
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999
        for i in range(1, len(front) - 1):
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                            obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                       obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])
    # print('track---distance : ',distance)
    return distance

# 根据非支配排序进行选择
def selection(population_size,target_population, front, chroms_obj_record):
    N = 0
    new_pop = []
    while N < population_size:
        for i in range(len(front)):
            N = N + len(front[i])
            if N > population_size:
                distance = calculate_crowding_distance(front[i], chroms_obj_record)
                sorted_cdf = sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop) == population_size:
                        break
                    new_pop.append(j)
                break
            else:
                new_pop.extend(front[i])

    population_list = []
    for n in new_pop:
        population_list.append(target_population[n])

    # print("tack----selection--new_pop len: ", len(new_pop))
    print("tack----selection--population_list len: ", len(population_list))
    return population_list, new_pop

# 交叉
# pop_after_sel ： 经过选择之后的gen
# 进行染色体交换
def crossover(pop_after_sel):
    temp = copy.deepcopy(pop_after_sel)
    pop_nextgen = temp
    pop_cross = []
    for i in range(0,len(temp),2):
        child_1, child_2 = pop_nextgen[i],pop_nextgen[i+1]
        # 进行gen一半一半的交换
        new_par1 = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_2)//2:]))
        new_par2 = np.concatenate((child_1[len(child_1) // 2:], child_2[:len(child_2) // 2]))
        pop_cross.append(new_par1)
        pop_cross.append(new_par2)
    # print("tack----crossover--pop_cross len: ", len(pop_cross))
    return pop_cross

# 变异
def mutation(pop_after_cross,mutation_rate,num_features):
    # 定义变异范围
    mutation_range = int(mutation_rate*num_features)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[n]
        rand_posi = []
        # 定义变异范围
        for i in range(0,mutation_range):
            pos = randint(0,num_features-1)
            rand_posi.append(pos)
        # 进行变异
        for j in rand_posi:
            chromosome[j] = not chromosome[j]
        # 将变异的结果添加到下一代中
        pop_next_gen.append(chromosome)
    # print("tack----mutation--pop_next_gen len: ", len(pop_next_gen))
    return pop_next_gen

# check 产生的target和chromo是否对应
def check_chromo(chromo):
    num_features = 0
    for c in chromo:
        if c :
            num_features = num_features + 1
    return num_features

# 产生后代
# num_generation : 产生后代数（进行多少次迭代）
# 以达到迭代次数就停止(停止条件)
# best_chromo ： 每一代最优的染色体
# best_score ： 最优染色体对应的分数
def generations(X_train,y_train,X_test,y_test,model,pop_size,num_features,drop_rate = 0.3,mutation_rate=0.3,num_generation=10):

    best_list = []
    best_obj = []
    population_nextgen = initilization_population(pop_size,num_features,drop_rate=drop_rate)
    font_0_num = 0
    # 进行generation循环
    for i in range(num_generation):
        # print(f"track --- generation_{i} -----")
        scores, pop_after_fit = fit_score(population=population_nextgen,
                                          model=model,
                                          X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_test,
                                          y_test=y_test)

        target_population, chroms_obj_record = get_targets(scores,pop_after_fit)

        front = non_dominated_sorting(population_size=pop_size,
                                      chroms_obj_record=chroms_obj_record)

        # print(f'track -- generation_{i}-front: ', front)
        # print(f'track -- generation_{i}-front len: ', len(front))

        pop_after_sel, new_pop= selection(population_size=pop_size,
                                          target_population=target_population,
                                          front=front,
                                          chroms_obj_record=chroms_obj_record
                                        )

        # print('track -- : pop_after_sel :', len(pop_after_sel))
        # print('track -- : new_pop :', len(new_pop))

        new_pop_obj = [chroms_obj_record[k] for k in new_pop]

        # print('track -- : new_pop_obj :', len(new_pop_obj))
        # print(f'track -- generation_{i}-new_pop[0]: ', new_pop[0])
        # print(f'track -- generation_{i}-target_population[new_pop[0]]: ', target_population[new_pop[0]])
        # print(f'track -- generation_{i}-pop_after_sel-: ',pop_after_sel[0])
        # print(f'track -- generation_{i}-new_pop_obj: ',new_pop_obj[0])
        # pop_after_sel 和 new_pop_obj 为重新排序的结果

        # 进行交叉
        # print('track -- 1: pop_after_sel :', len(pop_after_sel))
        pop_after_cross = crossover(pop_after_sel)
        # print('track -- 2: pop_after_sel :', len(pop_after_sel))

        # 进行变异
        population_nextgen = mutation(pop_after_cross,mutation_rate,num_features)
        # print('track --- : ',i)
        # 将每代generation最好的结果进行保存
        if i == 0:
            best_list = pop_after_sel
            best_obj = new_pop_obj
            # print('track -- ')
            # print('track -- : best_list :', len(best_list))
            # print('track -- : best_obj :', len(best_obj))
            plt.scatter(best_obj[0][1], best_obj[0][0], c='r')
            for index_best in range(1,len(best_obj)):
                plt.scatter(best_obj[index_best][1], best_obj[index_best][0], c='c')
            print(f'track---generation[{i}] best solution:')
            print(best_obj[0][1],round(float(best_obj[0][0]),4))
        else:
            total_list = copy.deepcopy(best_list) + copy.deepcopy(pop_after_sel)
            total_obj = copy.deepcopy(best_obj) + copy.deepcopy(new_pop_obj)

            # print('track -- :')
            # print('track -- : total_list[0] :',total_list[0])
            # print('track -- : check_chromo :', check_chromo(total_list[0]))
            # print('track -- : total_obj[0] :', total_obj[0])

            now_best_front = non_dominated_sorting(population_size=len(total_obj),chroms_obj_record=total_obj)
            best_list, best_pop = selection(population_size=len(total_obj),
                                            front=now_best_front,
                                            chroms_obj_record=total_obj,
                                            target_population=total_list)
            best_obj = [total_obj[k] for k in best_pop]

            # best_list 和 best_obj  是每次generation结束的最佳
            # 绘制散点图
            plt.figure(i)
            x_dim = []
            y_dim = []

            selected_best_solution = len(now_best_front[0]) + len(now_best_front[1])

            for j in range(selected_best_solution,len(best_obj)):
                x_dim.append(best_obj[j][1])
                y_dim.append(best_obj[j][0])
            plt.scatter(x_dim, y_dim, c='c')
            # 绘制pareto front 0
            for index_best_1 in range(len(now_best_front[0])):
                plt.scatter(best_obj[index_best_1][1], best_obj[index_best_1][0], c='r')
            print(f'track---generation[{i}] best solution:')
            print(best_obj[0][1],round(float(best_obj[0][0]),4))
            # # 绘制pareto front 1
            # for index_best_2 in range(len(now_best_front[0]),len(now_best_front[1])):
            #     plt.scatter(best_obj[index_best_2][1], best_obj[index_best_2][0], c='b')
            # 保存最后的front[0]的解决方案个数
            if num_generation-1 == i:
                font_0_num = len(now_best_front[0])

        plt.xlim(0, 35)
        plt.ylim(0, 0.5)
        plt.xlabel("Number of Selected Features",fontsize=18)
        plt.ylabel("Error Distance (1 - Accuracy)",fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.title("Two Targets Optimization")
        plt.savefig(f'../ga-pop32-gen50/generation_{i}.png',dpi=600)
        # plt.show()

    # print('track -- ')
    # print('track -- : best_list :', best_list[0])
    # print('track -- : check_chromo :', check_chromo(best_list[0]))
    # print('track -- : best_obj :', best_obj[0])
    top_selected_features, top_object = get_top_solution(best_list,best_obj,font_0_num)


    return top_selected_features, top_object,font_0_num


def get_top_solution(best_list,best_obj,num_solutions):
    # 获取最优选择的前num_solutions组特征index
    top_num = num_solutions
    if top_num <= len(best_list):
        top_selected_features = []
        top_object = []
        for t in range(top_num):
            index_count = 0
            features_index = []
            for c in best_list[t]:
                if c:
                    features_index.append(index_count)
                index_count += 1
            top_selected_features.append(features_index)
            top_object.append(best_obj[t])
    else:
        top_selected_features = []
        top_object = []

    return top_selected_features,top_object

# 产生后代
# num_generation : 产生后代数（进行多少次迭代）
# 以达到迭代次数就停止(停止条件)
# best_chromo ： 每一代最优的染色体
# best_score ： 最优染色体对应的分数
def generations_v2(X_train,X_train_val,y_train,y_train_val,X_test,y_test,model,pop_size,num_features,drop_rate = 0.3,mutation_rate=0.3,num_generation=10,name=None):

    best_list = []
    best_obj = []
    population_nextgen = initilization_population(pop_size,num_features,drop_rate=drop_rate)
    font_0_num = 0
    best_solution_gen = []
    # 进行generation循环
    for i in range(num_generation):
        plt.figure(figsize=(7,6))
        # print(f"track --- generation_{i} -----")
        scores, pop_after_fit = fit_score(population=population_nextgen,
                                          model=model,
                                          X_train=X_train,
                                          X_train_val=X_train_val,
                                          y_train=y_train,
                                          y_train_val=y_train_val,
                                          X_test=X_test,
                                          y_test=y_test)

        target_population, chroms_obj_record = get_targets(scores,pop_after_fit)

        front = non_dominated_sorting(population_size=pop_size,
                                      chroms_obj_record=chroms_obj_record)

        # print(f'track -- generation_{i}-front: ', front)
        # print(f'track -- generation_{i}-front len: ', len(front))

        pop_after_sel, new_pop= selection(population_size=pop_size,
                                          target_population=target_population,
                                          front=front,
                                          chroms_obj_record=chroms_obj_record
                                        )

        # print('track -- : pop_after_sel :', len(pop_after_sel))
        # print('track -- : new_pop :', len(new_pop))

        new_pop_obj = [chroms_obj_record[k] for k in new_pop]

        # print('track -- : new_pop_obj :', len(new_pop_obj))
        # print(f'track -- generation_{i}-new_pop[0]: ', new_pop[0])
        # print(f'track -- generation_{i}-target_population[new_pop[0]]: ', target_population[new_pop[0]])
        # print(f'track -- generation_{i}-pop_after_sel-: ',pop_after_sel[0])
        # print(f'track -- generation_{i}-new_pop_obj: ',new_pop_obj[0])
        # pop_after_sel 和 new_pop_obj 为重新排序的结果

        # 进行交叉
        # print('track -- 1: pop_after_sel :', len(pop_after_sel))
        pop_after_cross = crossover(pop_after_sel)
        # print('track -- 2: pop_after_sel :', len(pop_after_sel))

        # 进行变异
        population_nextgen = mutation(pop_after_cross,mutation_rate,num_features)
        # print('track --- : ',i)
        # 将每代generation最好的结果进行保存
        if i == 0:
            best_list = pop_after_sel
            best_obj = new_pop_obj
            # print('track -- ')
            # print('track -- : best_list :', len(best_list))
            # print('track -- : best_obj :', len(best_obj))
            plt.scatter(best_obj[0][1], best_obj[0][0], c='r')
            for index_best in range(1,len(best_obj)):
                plt.scatter(best_obj[index_best][1], best_obj[index_best][0], c='c')
            # print(f'track---generation[{i}] best solution:')
            # print(best_obj[0][1],round(float(best_obj[0][0]),4))

            # sort_selected_gen_model(top_object=best_obj,
            #                         error_toler=0.005)

        else:
            total_list = copy.deepcopy(best_list) + copy.deepcopy(pop_after_sel)
            total_obj = copy.deepcopy(best_obj) + copy.deepcopy(new_pop_obj)

            # print('track -- :')
            # print('track -- : total_list[0] :',total_list[0])
            # print('track -- : check_chromo :', check_chromo(total_list[0]))
            # print('track -- : total_obj[0] :', total_obj[0])

            now_best_front = non_dominated_sorting(population_size=len(total_obj),chroms_obj_record=total_obj)
            best_list, best_pop = selection(population_size=len(total_obj),
                                            front=now_best_front,
                                            chroms_obj_record=total_obj,
                                            target_population=total_list)
            best_obj = [total_obj[k] for k in best_pop]

            # best_list 和 best_obj  是每次generation结束的最佳
            # 绘制散点图
            plt.figure(i,figsize=(8,8))
            x_dim = []
            y_dim = []

            selected_best_solution = len(now_best_front[0]) + len(now_best_front[1])

            for j in range(selected_best_solution,len(best_obj)):
                x_dim.append(best_obj[j][1])
                y_dim.append(best_obj[j][0])
            plt.scatter(x_dim, y_dim, c='c')
            # 绘制pareto front 0
            pareto_optimal_set = []
            for index_best_1 in range(len(now_best_front[0])):
                pareto_optimal_set.append(best_obj[index_best_1])
                plt.scatter(best_obj[index_best_1][1], best_obj[index_best_1][0], c='r')
            print(f'track---generation[{i}] best solution:')
            # print(best_obj[0][1],round(float(best_obj[0][0]),4))
            best_solution_gen = sort_selected_gen_model(
                                    top_object=pareto_optimal_set,
                                    error_toler=0.005
            )
            if num_generation-1 == i:
                font_0_num = len(now_best_front[0])
            # evaluate EScore

        # 保存每代最佳rs方案的目标值和rs值
        # best_rs_gen.append(best_solution_gen)

        # 绘制每代的Pareto目标空间
        plt.xlim(0, 35)
        plt.ylim(0, 0.5)
        plt.xlabel("The number of Features",fontsize=15)
        plt.ylabel("Error Distance(1 - f1 score)",fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.title("Two Targets Optimization")
        plt.savefig(f'../{name}_gaImages/generation_{i+1}.png',dpi=600)
        # plt.show()

    # print('track -- ')
    # print('track -- : best_list :', best_list[0])
    # print('track -- : check_chromo :', check_chromo(best_list[0]))
    # print('track -- : best_obj :', best_obj[0])
    top_selected_features, top_object = get_top_solution(best_list,best_obj,font_0_num)


    return top_selected_features, top_object,font_0_num

def sort_selected_final_model(features,columns,top_object,front_0_num,error_toler):
    print("==================================")
    # custom selected
    EScores = []
    # sort (based on the error distance)
    sort_result = sorted(top_object, key = lambda top_object:(top_object[0]))
    ob1_error = sort_result[0][0]
    ob1_num = sort_result[0][1]
    for i in range(1,len(sort_result)): # modified
        ob_error = sort_result[i][0]
        ob_num = sort_result[i][1]

        weight = math.log(ob1_num-ob_num+1)
        tempScore = weight*((ob1_num-ob_num)*error_toler - abs(ob1_error-ob_error))*100
        EScores.append([ob_error,ob_num,tempScore])

    print('track --- sort_result : ',sort_result)

    # ==================================
    # 比较rs = 0 和 算出来最高RS的模型
    if(EScores != []):
        # 按rs分数排序
        sort_rs = sorted(EScores, key=lambda EScores: (EScores[2]),reverse=True)
        print('track --- Sort RS : ', sort_rs)
        if(sort_rs[0][2] >= 0.0):
            best_model_selected =  [sort_rs[0][0],sort_rs[0][1]]
        else:
            best_model_selected = sort_result[0]
    else:
        best_model_selected = sort_result[0]
        print('track --- Sort RS : ',EScores)
    print("track--selected best model: ",best_model_selected)
    # ==================================
    print("==================================")
    return best_model_selected

def sort_selected_gen_model(top_object,error_toler):

    # custom selected
    EScores = []
    Selected_S = []
    # sort (based on the error distance)
    sort_result = sorted(top_object, key = lambda top_object:(top_object[0]))
    ob1_error = sort_result[0][0]
    ob1_num = sort_result[0][1]
    # add baseline (solution1 )
    Selected_S.append([0,ob1_error,ob1_num,0])

    for i in range(1,len(sort_result)):
        ob_error = sort_result[i][0]
        ob_num = sort_result[i][1]

        weight = math.log(ob1_num-ob_num+1)
        tempScore = weight*((ob1_num-ob_num)*error_toler - abs(ob1_error-ob_error))*100
        EScores.append([ob_error,ob_num,tempScore])
        # add higher EScore
        if tempScore > 0:
            Selected_S.append([i,ob_error,ob_num,tempScore])

    calculate_results = sorted(Selected_S, key = lambda Selected_S:(Selected_S[3]),reverse=True)
    # print 打印结果
    print(calculate_results)
    # 选出每代最佳模型 calculate_results[i][3] : rs分数

    best_solution_gen = calculate_results[0]

    return best_solution_gen
'''
函数：绘制评估指标图
'''
def plot_metrics(num_obj,soulations_results,len_features):
    # ### 绘制所有模型的预测结评估果
    n_groups = num_obj
    fig, ax = plt.subplots(figsize=(8,6))
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity1 = 0.2
    opacity2 = 0.4
    opacity3 = 0.6
    opacity4 = 0.8
    opacity5 = 0.9
    # colors = ['powderblue','lightskyblue','mediumblue','cornflowerblue','royalblue']
    colors = ['royalblue','royalblue','royalblue','royalblue','royalblue']
    solution_1 = soulations_results['solution_1']
    solution_2 = soulations_results['solution_2']
    solution_3 = soulations_results['solution_3']
    solution_4 = soulations_results['solution_4']
    solution_5 = soulations_results['solution_5']

    lables = []
    for i in range(num_obj):

        lables.append(f'solution_{i+1}({len_features[i]})')

    rects1 = plt.bar(
        index,
        solution_1,
        bar_width,
        alpha=opacity1,
        color=colors[0],
        label=lables[0]
    )

    rects2 = plt.bar(
        index+bar_width,
        solution_2,
        bar_width,
        alpha=opacity2,
        color=colors[1],
        label=lables[1]
    )


    rects3 = plt.bar(
        index+2*bar_width,
        solution_3,
        bar_width,
        alpha=opacity3,
        color=colors[2],
        label=lables[2]
    )


    rects4 = plt.bar(
        index+3*bar_width,
        solution_4,
        bar_width,
        alpha=opacity4,
        color=colors[3],
        label=lables[3]
    )

    rects5 = plt.bar(
        index+4*bar_width,
        solution_5,
        bar_width,
        alpha=opacity5,
        color=colors[4],
        label=lables[4]
    )


    plt.ylim(0,1)
    plt.title('Performance',fontsize=18)
    plt.xticks(index+bar_width,('Accuracy','Precision','Recall','F1 Score','AUC'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='lower right',fontsize=15)
    plt.tight_layout()
    plt.savefig('../images/5_solutions_metrics_1.png')
    plt.show()

