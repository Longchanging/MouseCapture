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

rcParams['figure.figsize'] = 12, 4
warnings.filterwarnings("ignore")

def upsample(df):
    
    from sklearn.utils import resample

    #  分离多数和少数类别
    df_majority = df[df.label == 1]
    df_minority = df[df.label == 0]
    
    # 上采样少数类别
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results
    
    # 合并多数类别同上采样过的少数类别
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df_upsampled.label.value_counts()
    
    return df_upsampled


if __name__ == '__main__':
        
    # 读 数据集,命名
    train_data = pd.read_csv(train_feature, engine='python' , delimiter=' ')
    columns_name = []
    for i in range(len(train_data.iloc[0]) - 1):
        columns_name.append('feature_' + str(i + 1))
    columns_name.append('label')
    train_data.columns = columns_name
    
    # 重采样
    train_data = upsample(train_data)
    
    train_feature = train_data.iloc[:, :-1]
    label = train_data.iloc[:, -1].astype(int)
    test_feature = pd.read_csv(test_feature, engine='python' , delimiter=' ')
    print(train_feature.describe())
    print(label.describe())
    print(test_feature.describe())
    
    # xgb.train()
    
    gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=50, min_samples_split=50, max_depth=9,
                                     min_samples_leaf=4, max_features=15, subsample=0.7)

    # train
    print('Start training...')
    gbm.fit(train_feature, label)
    label_pred = gbm.predict(train_feature)
    label_predprob = gbm.predict_proba(train_feature)[:, 1]
    print ("Accuracy: %.4f" % metrics.accuracy_score(label, label_pred))
    print ("AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob))
    
    # cross-validation 交叉验证
    cv_score = cross_val_score(gbm, train_feature, label, cv=5, scoring='roc_auc')
    print ("CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # predict
    print('Start predicting...')
    y_pred = gbm.predict(test_feature)
    y_predprob = gbm.predict_proba(test_feature)[:, 1]

    from collections import Counter
    # 观察一下y分布
    print ('predict label count for A:')
    print(Counter(y_pred))
