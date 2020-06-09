# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:09:35 2020

@author: mohammad.asif
"""


import pandas as pd 
import numpy as np                     # For mathematical calculations 
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


train=pd.read_csv("E:\DataScience\DeploymentLogistic\LoanApprovaltrain.csv.csv") 
train.shape
test=pd.read_csv("E:\DataScience\DeploymentLogistic\LoanApprovaltest.csv")
train_original=train.copy() 
test_original=test.copy()
# Data Analysis
train.shape
test.shape

train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts().plot.bar()


plt.figure(1) 
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()



# Null value check and fill

train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()
train.isnull().any()

test.isnull().sum()

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().any()
train.dtypes
test.dtypes

#Lets drop the Loan_ID variable as it do not have any effect on the loan status. We will do the same changes to the test dataset which we did for the training dataset.
train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)

# encoding : convert the cat. columns to numerica one and converting to float

train.dtypes
test.dtypes
enc = LabelEncoder()
train['Gender'] = enc.fit_transform(train['Gender'])
train['Married'] = enc.fit_transform(train['Married'])
train['Dependents'] = enc.fit_transform(train['Dependents'])
train['Education'] = enc.fit_transform(train['Education'])
train['Self_Employed'] = enc.fit_transform(train['Self_Employed'])
train['Property_Area'] = enc.fit_transform(train['Property_Area'])
train['Loan_Status'] = enc.fit_transform(train['Loan_Status'])
train.dtypes
train['Gender']=train['Gender'].astype(float)
train['Married']=train['Married'].astype(float)
train['Dependents']=train['Dependents'].astype(float)
train['Education']=train['Education'].astype(float)
train['Self_Employed']=train['Self_Employed'].astype(float)
train['Property_Area']=train['Property_Area'].astype(float)
train['Loan_Status']=train['Loan_Status'].astype(float)
train['ApplicantIncome']=train['ApplicantIncome'].astype(float)
train.dtypes

test.dtypes
test['Gender'] = enc.fit_transform(test['Gender'])
test['Married'] = enc.fit_transform(test['Married'])
test['Dependents'] = enc.fit_transform(test['Dependents'])
test['Education'] = enc.fit_transform(test['Education'])
test['Self_Employed'] = enc.fit_transform(test['Self_Employed'])
test['Property_Area'] = enc.fit_transform(test['Property_Area'])
#test['Loan_Status'] = enc.fit_transform(test['Loan_Status'])
test['Gender']=test['Gender'].astype(float)
test['Married']=test['Married'].astype(float)
test['Dependents']=test['Dependents'].astype(float)
test['Education']=test['Education'].astype(float)
test['Self_Employed']=test['Self_Employed'].astype(float)
test['Property_Area']=test['Property_Area'].astype(float)
#test['Loan_Status']=test['Loan_Status'].astype(float)
test['ApplicantIncome']=test['ApplicantIncome'].astype(float)
test['CoapplicantIncome']=test['CoapplicantIncome'].astype(float)


X = train.drop('Loan_Status',1) 
y = train.Loan_Status
X.isnull().any()
y.isnull().any()
X.shape
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.3)
classifier = LogisticRegression() 
classifier.fit(x_train, y_train)
classifier.score(x_test,y_test)

pk.dump(classifier,open('model.pkl','wb'))
model = pk.load(open('model.pkl','rb'))






#pred_test = model.predict(test)
#submission=pd.read_csv("E:\DataScience\sample_submission.csv")
#submission['Loan_Status']=pred_test 
#submission['Loan_ID']=test_original['Loan_ID']
#submission['Gender']=test_original['Gender']
#submission['ApplicantIncome']=test_original['ApplicantIncome']
#submission['Loan_Status'].replace(0, 'N',inplace=True) 
#submission['Loan_Status'].replace(1, 'Y',inplace=True)
#pd.DataFrame(submission, columns=['Loan_ID','Loan_Status','Gender','ApplicantIncome']).to_csv('E:\DataScience\PredictedHomeloan.csv')




