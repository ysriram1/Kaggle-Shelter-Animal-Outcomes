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
from sklearn.preprocessing import MinMaxScaler

zscore_fit = StandardScaler().fit(train)
train_normalized = zscore_fit.transform(train)
test_normalized = zscore_fit.transform(test)

minmax_fit = MinMaxScaler().fit(train)
train_standardized = minmax_fit.transform(train)
test_standardized = minmax_fit.transform(test)

from sklearn.cross_validation import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(train_standardized, target, test_size=0.33, random_state=42)

########################First try to just get the accuracy values -- this will indicate what method makes sense#####################
from sklearn.ensemble import AdaBoostClassifier as ad
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.naive_bayes import GaussianNB
from sknn.mlp import Classifier, Layer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


nn = Classifier(
    layers=[Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
nn.fit(train_normalized, target)
y_predict = nn.predict(test_normalized)

ad_fit = ad(n_estimators = 10).fit(X_train,y_train)
y_pred = ad_fit.predict(X_test)
ad_acc = accuracy_score(y_pred, y_test) #0.60

y_pred = rf().fit(X_train,y_train).predict(X_test)
rf_acc = accuracy_score(y_pred, y_test) #0.59

gnb = GaussianNB() 
y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb_acc = accuracy_score(y_pred, y_test) #0.075 (extremely low)

svc = svm.SVC()
svc = svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_pred, y_test)


ldaC = LDA().fit(X_train, y_train)
y_pred = ldaC.predict(X_test)
lda_acc = accuracy_score(y_pred, y_test) #0.62

confusionMat=confusion_matrix(y_pred, y_test)

recalls = []
totals = []
predictionCounts = []

for i in range(len(confusionMat)):
    recalls.append(float(confusionMat[i,i])/np.sum(confusionMat[i]))
    totals.append(np.sum(confusionMat[:,i]))
    predictionCounts.append(np.sum(confusionMat[i]))


logreg = LogisticRegression().fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)
log_acc =  accuracy_score(y_pred, y_test) #0.63 highest

clf = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf_acc =  accuracy_score(y_pred, y_test) #0.61 

neigh = KNeighborsClassifier(n_neighbors=13).fit(X_train, y_train)
y_pred = neigh.predict(X_test)
nn_acc =  accuracy_score(y_pred, y_test) #0.61


#########################################
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

def calc_params(X, y, clf, param_values, param_name, K, metric = 'accuracy'):
    '''This function takes the classfier, the training data and labels, the name of the
    parameter to vary, a list of values to vary by, and a number of folds needed for 
    cross validation and returns a the test and train scores (accuracy or recall) and also
    prints out a graph that shows the variation of the these scores accross different 
    pararmeter values.
    '''
    # Convert input to Numpy arrays
    X = np.array(X)
    y = np.array(y)

    # initialize training and testing scores with zeros
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))
    
    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):
        #print(param_name, ' = ', param_value)
        
        # set classifier parameters
        clf.set_params(**{param_name:param_value})
        
        # initialize the K scores obtained for each fold
        k_train_scores = np.zeros(K)
        k_test_scores = np.zeros(K)
        
        # create KFold cross validation or stratified bootstrap validation
        if metric == 'accuracy':
            cv = KFold(len(X), K, shuffle=True, random_state=99)
        
        if metric == 'recall':
            cv = StratifiedKFold(y, n_folds = K, shuffle=True, random_state=99)
        
        # iterate over the K folds
        for j, (train, test) in enumerate(cv):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets
            clf.fit([X[k] for k in train], y[train])
            if metric == 'accuracy':
                k_train_scores[j] = clf.score([X[k] for k in train], y[train])
                k_test_scores[j] = clf.score([X[k] for k in test], y[test])
            elif metric == 'recall':
                fit = clf.fit(X[train],y[train])
                k_train_scores[j] = recall_score(fit.predict(X[train]),y[train], pos_label=1, average = 'binary')
                k_test_scores[j] = recall_score(fit.predict(X[test]),y[test], pos_label=1, average = 'binary')
       
       # store the mean of the K fold scores
        if metric == 'accuracy':
            train_scores[i] = np.mean(k_train_scores)
            test_scores[i] = np.mean(k_test_scores)
        if metric == 'recall':
            train_scores[i] = np.mean(k_train_scores)
            test_scores[i] = np.mean(k_test_scores)
       
    # plot the training and testing scores in a log scale
    plt.close()
    plt.figure()
    plt.plot(param_values, train_scores, label='Train', alpha=0.4, lw=2, c='b')
    plt.plot(param_values, test_scores, label='Test', alpha=0.4, lw=2, c='g')
    plt.legend(loc=7)
    plt.xlabel(param_name + " values")
    if metric == 'accuracy': plt.ylabel("Mean cross validation accuracy")
    if metric == 'recall': plt.ylabel("Mean cross validation recall (Senstivity) for label 1")
    plt.show()

    # return the training and testing scores on each parameter value
    return train_scores, test_scores


########FOCUSING ON LOGISTIC REGRESSION AND LDA TEST DATA########################
from sklearn.linear_model import LogisticRegressionCV

logregCV = LogisticRegressionCV(cv= 10, solver = 'lbfgs', penalty = 'l2').fit(train_standardized, target)
logCV_acc = logregCV.scores_
y_pred = logregCV.predict_proba(test_standardized)


ldaC = LDA().fit(train_standardized, target)
y_pred = ldaC.predict_proba(test_standardized)

ad_fit = ad(n_estimators = 10).fit(train_standardized, target)
y_pred = ad_fit.predict_proba(test_standardized)


rf_fit = rf(random_state=99).fit(train_standardized, target)

splitSizes = list(range(1,10,1))
train_scores, test_scores = calc_params(train_standardized, target, rf_fit, splitSizes, 'min_samples_leaf', 5, metric = 'accuracy')
pd.DataFrame(np.array([test_scores, splitSizes]).T, columns = ['Test Recall', 'Minimum Split Size'])

nEst = range(1, 51, 10)
train_scores, test_scores = calc_params(train_standardized, target, rf_fit, nEst, 'n_estimators', 5, metric = 'accuracy')
pd.DataFrame(np.array([test_scores, nEst]).T, columns = ['Test Recall', 'Number of Estimators'])

rf_fit = rf(min_samples_leaf=8, n_estimators = 1).fit(train_standardized, target) #min split of 8 gives least variance
y_pred = rf_fit.predict_proba(test_standardized) 

################SUBMISSION#################

pred_df = pd.DataFrame(y_pred); #submission_format = pd.get_dummies(pred_df); 
pred_df.columns = ['Adoption','Died','Euthanasia','Return_to_owner','Transfer']
pred_df.index += 1
pred_df.index.rename('ID', inplace=True)
pred_df.to_csv('./submission.csv')