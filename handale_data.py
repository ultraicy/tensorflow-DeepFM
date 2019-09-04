#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:23:21 2019

@author: nickwu
"""

import pandas as pd
df1 = pd.read_csv('adult.data',header=None,names = ['age','workclass','fnlwgt','educate','education-num',\
                                        'marital-status','occupation','relationship','race',\
                                        'sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])


dict1 = {'<=50K':0,'>50K':1}
def get_label(x):
    if x==' <=50K':
        return 0
    else:
        return 1
df1['target'] = df1['salary'].map(get_label)

df1.drop(['salary'],axis=1 ,inplace=True)
df1= df1.reset_index()

from sklearn.model_selection import train_test_split


X_train, X_test = train_test_split(df1)

X_train.to_csv('data/train_data.csv',index=False)
X_test.to_csv('data/test_data.csv',index=False)
