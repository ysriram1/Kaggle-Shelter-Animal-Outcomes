# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/syarlag1/Desktop/kaggle animal shelter')    

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

train = train_raw.iloc[:,1:train_raw.shape[1]]
target = train_raw.iloc[:,0]

test = test_raw


## First try is without train test split

from sklearn.ensemble import RandomForestClassifier as rf

fit = rf(random_state=99).fit(train,target)