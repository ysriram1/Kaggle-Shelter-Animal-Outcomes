# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/syarlag1/Desktop/Kaggle-Shelter-Animal-Outcomes')    

train_raw = pd.read_csv('./train.csv', usecols=[3,5,6,7,8,9])
test_raw = pd.read_csv('./test.csv', usecols=range(3,8))



#replacing the Age column with the numerical value in days. '1 Year' to 365.

age_numerical = []

for age in train_raw.AgeuponOutcome: 
    if age is np.nan: 
        age_numerical.append(None)
        continue
    i = 2
    if age[1].isdigit(): i = 3
    if age[i] == 'd': age_numerical.append(int(age[:i-1]))
    elif age[i] == 'w': age_numerical.append(int(age[:i-1])*7)
    elif age[i] == 'm': age_numerical.append(int(age[:i-1])*30)
    elif age[i] == 'y': age_numerical.append(int(age[:i-1])*365)

train_raw.AgeuponOutcome = pd.Series(age_numerical)

age_numerical = []

for age in test_raw.AgeuponOutcome:
    if age is np.nan: 
        age_numerical.append(None)
        continue
    i = 2
    if age[1].isdigit(): i = 3
    if age[i] == 'd': age_numerical.append(int(age[:i-1]))
    elif age[i] == 'w': age_numerical.append(int(age[:i-1])*7)
    elif age[i] == 'm': age_numerical.append(int(age[:i-1])*30)
    elif age[i] == 'y': age_numerical.append(int(age[:i-1])*365)
        
test_raw.AgeuponOutcome = pd.Series(age_numerical)     

target = train_raw.iloc[:,0]
train_raw = train_raw.iloc[:,1:train_raw.shape[1]]

train_raw.replace('[/,\s]','',inplace=True, regex=True)
test_raw.replace('[/,\s]','',inplace=True, regex=True)

train_raw.replace([np.inf,np.nan],0,inplace=True)
test_raw.replace([np.inf,np.nan],0,inplace=True)


combined_raw = train_raw.append(test_raw)


combined_raw_encoded = pd.get_dummies(combined_raw)

train = combined_raw_encoded.iloc[:train_raw.shape[0],:]
test = combined_raw_encoded.iloc[train_raw.shape[0]:combined_raw.shape[0],:]

from sklearn.preprocessing import StandardScaler

zscore_fit = StandardScaler().fit(train)
train_normalized = zscore_fit.transform(train)
test_normalized = zscore_fit.transform(test)

from sklearn.cross_validation import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)

from sklearn.metrics import accuracy_score, confusion_matrix

########################First try is without train test split#####################
from sklearn.ensemble import AdaBoostClassifier as ad
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.naive_bayes import GaussianNB
from sknn.mlp import Classifier, Layer
from sklearn import svm

nn = Classifier(
    layers=[Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
nn.fit(train_normalized, target)
y_predict = nn.predict(test_normalized)

ad_fit = ad(n_estimators = 10).fit(train,target)
y_pred = ad_fit.predict(test)

y_pred = rf().fit(train,target).predict(test)

gnb = GaussianNB()
y_pred = gnb.fit(train, target).predict(test)


clf = svm.SVC()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


################SUBMISSION#################

pred_df = pd.DataFrame(y_pred); submission_format = pd.get_dummies(pred_df); 
submission_format.columns = ['Adoption','Died','Euthanasia','Return_to_owner','Transfer']
submission_format.to_csv('./submission.csv')