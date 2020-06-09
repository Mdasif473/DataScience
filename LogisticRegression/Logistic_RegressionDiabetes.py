# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:09:36 2020

@author: mohammad.asif
"""


import pandas as pd
import numpy as np                 # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
#%matplotlib inline 
import warnings# To ignore any warnings warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import pickle as pk
from sklearn import metrics


train=pd.read_csv("E:\DataScience\diabetes.csv")
train.isnull().any()
train.dtypes
train['Pregnancies']=train['Pregnancies'].astype(float)
train['Glucose']=train['Glucose'].astype(float)
train['BloodPressure']=train['BloodPressure'].astype(float)
train['SkinThickness']=train['SkinThickness'].astype(float)
train['Insulin']=train['Insulin'].astype(float)
train['Age']=train['Age'].astype(float)
train['Outcome']=train['Outcome'].astype(float)

X=train.drop('Outcome',1)
y=train.Outcome

X.isnull().any()
y.isnull().any()
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.3)
model = LogisticRegression() 
model.fit(x_train, y_train)
model.score(x_test,y_test)
y_pred=model.predict(x_test)
   
cnf_met = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy Score: ",metrics.accuracy_score(y_test,y_pred))
