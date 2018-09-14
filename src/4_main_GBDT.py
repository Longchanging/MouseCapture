# coding:utf-8
import gc
import os
import warnings

import lightgbm
import matplotlib
from matplotlib.pylab import rcParams
from sklearn import metrics
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb


train_feature = '../data/train_feature.txt'
test_feature = '../data/test_feature.txt'
result_saver_normal = '../data/results.csv'
result_saver_upsampled = '../data/results_sampled.csv'
cross_validation_K = 10
use_upsample_or_not = True
test_classifiers = ['KNN', 'DT', 'SVM', 'GBDT']
predict_samples = 10000
base_gini_rate = 2600 / 400

rcParams['figure.figsize'] = 12, 4
warnings.filterwarnings("ignore")

def naive_bayes_classifier(trainX, trainY):  # Multi nomial Naive Bayes Classifier
    
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(trainX, trainY)
    
    return model

def knn_classifier(trainX, trainY):  # KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(trainX, trainY)
    return model

def logistic_regression_classifier(trainX, trainY):  # Logistic Regression Classifier
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(trainX, trainY)
    
    return model

def random_forest_classifier(trainX, trainY):  # Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=30)
    model.fit(trainX, trainY)
    
    return model

def decision_tree_classifier(trainX, trainY):  # Decision Tree Classifier
    
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(trainX, trainY)
    
    return model
 
def gradient_boosting_classifier(trainX, trainY):  # GBDT(Gradient Boosting Decision Tree) Classifier
    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(trainX, trainY)
    
    return model

def ada_boosting_classifier(trainX, trainY):  # AdaBoost Ensemble Classifier
    
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(trainX, trainY)
    
    return model

def svm_classifier(trainX, trainY):  # SVM Classifier

    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(trainX, trainY)
    
    return model

def gbm_classifier(trainX, trainY):
    gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=50, min_samples_split=50, max_depth=9,
                                     min_samples_leaf=4, max_features=15, subsample=0.7)
    gbm.fit(X_train, y_train)
    return gbm

def upsample(df, use_upsample_or_not):
    
    df_majority = df[df.label == 1]
    # df_majority.label = 0
    df_minority = df[df.label == 0]
    # df_minority.label = 1
    
    if use_upsample_or_not:
        print('Up sampling used')
        from sklearn.utils import resample
            
        # 上采样少数类别
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority),  # to match majority class
                                         random_state=123)  # reproducible results
        
        # 合并多数类别同上采样过的少数类别
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        df_upsampled.label.value_counts()
        
        return df_upsampled
    
    else:
        return df
    
def calculate_distances(list_1, list_2):
    from collections import Counter
    d = dict(Counter(list_1))
    if d[1] == len(list_1):
        d[0] = 1
    d_0 = d[0] / (d[0] + d[1])
    d_1 = d[1] / (d[0] + d[1])
    e = dict(Counter(list_2))
    if e[1] == len(list_2):
        e[0] = 1
    e_0 = e[0] / (e[0] + e[1])
    e_1 = e[1] / (e[0] + e[1])
    distance_ = (d_0 - e_0) * (d_0 - e_0) + (d_1 - e_1) * (d_1 - e_1)
    return distance_ / 2

def cal_distance(list_A):
    from collections import Counter
    d = dict(Counter(list_A))
    if len(d) < 2:
        if 0 in d.keys():
            d[1] = 1
        else:
            d[0] = 1
    absolute_gini = d[1] / d[0] if d[1] > d[0] else d[0] / d[1]
    Relative_gini = absolute_gini / base_gini_rate
    return Relative_gini

if __name__ == '__main__':
        
    # 读 数据集,命名
    train_data = pd.read_csv(train_feature, engine='python' , delimiter=' ')
    columns_name = []
    for i in range(len(train_data.iloc[0]) - 1):
        columns_name.append('feature_' + str(i + 1))
    columns_name.append('label')
    train_data.columns = columns_name
            
    # 重采样
    train_data = upsample(train_data, use_upsample_or_not)
    
    # 读取预测集数据
    train_feature = train_data.iloc[:, :-1]
    label = train_data.iloc[:, -1].astype(int)
    test_feature = pd.read_csv(test_feature, engine='python' , delimiter=' ', skiprows=90000)
    print(train_feature.describe())
    print(label.describe())
    print(test_feature.describe())
    
    # 在切分后的数据上 切分 训练集和测试集
    # test_ratio = 0.2
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(train_feature, label, \
    X_train, y_train = train_feature, label
    
    # 选择使用的特征，随机选取
    
    X_train = X_train  # .iloc[:, 20:30]
    test_feature = test_feature  # .iloc[:, 20:30]
    
    # 归一化
    #     from sklearn import preprocessing
    #     max_min_model = preprocessing.MinMaxScaler().fit(X_train)
    #     X_train = max_min_model.transform(X_train) 
    

    # 定义使用的算法
    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                 'GBDT':gradient_boosting_classifier,
                 'AdaBoost':ada_boosting_classifier,
                 'gbm':gbm_classifier,
    }
    
    ##### 使用不同算法做训练和预测
    import os
    
    if use_upsample_or_not:
        file_ = result_saver_upsampled
    else:
        file_ = result_saver_normal
        
    if os.path.exists(file_):
        os.remove(file_)
    fid = open(file_, 'a')

    str_name = 'Classifier,Accuracy,Recall,ROC,F1,Gini\n'
    fid.write(str_name)

    for classifier in test_classifiers:
        
        print('\nAlgorithm:%s' % classifier)
        
        from sklearn.metrics import fbeta_score, make_scorer
        ftwo_scorer = make_scorer(fbeta_score, beta=5)

        # train
        model = classifiers[classifier](X_train, y_train)
                
        # cross-validation 交叉验证
        accuracy_score = cross_val_score(model, X_train, y_train, cv=cross_validation_K, scoring='accuracy')
        recall_score = cross_val_score(model, X_train, y_train, cv=cross_validation_K, scoring='recall')
        f1_score = cross_val_score(model, X_train, y_train, cv=cross_validation_K, scoring=ftwo_scorer)
        roc_auc_score = cross_val_score(model, X_train, y_train, cv=cross_validation_K, scoring='roc_auc')
        print ("Accuracy score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
        np.mean(accuracy_score), np.std(accuracy_score), np.min(accuracy_score), np.max(accuracy_score)))
        print ("recall score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
        np.mean(recall_score), np.std(recall_score), np.min(recall_score), np.max(recall_score)))
        print ("weighted f1 score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
        np.mean(f1_score), np.std(f1_score), np.min(f1_score), np.max(f1_score)))
        print ("roc_auc score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
        np.mean(roc_auc_score), np.std(roc_auc_score), np.min(roc_auc_score), np.max(roc_auc_score)))
        
        # 预测集上的距离
        # predict
        print('Start predicting...')
        y_pred = model.predict(test_feature)
        y_predprob = model.predict_proba(test_feature)[:, 1]
        from collections import Counter
        # 观察一下y分布
        print ('predict label count for A:')
        print(Counter(y_pred))
        # true_distance = calculate_distances(y_pred, y_train)
        true_distance = cal_distance(y_pred)
        print('distances:%f' % true_distance)
        
        str_values = '%s,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (classifier, np.mean(accuracy_score), \
                    np.mean(recall_score), np.mean(f1_score), np.mean(roc_auc_score), true_distance)
        fid.write(str_values)
