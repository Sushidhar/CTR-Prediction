# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 20:35:29 2017

@author: Sushidhar
"""

import csv, sqlite3
import odo
import pandas as pd
import seaborn as sns
import blaze as bz
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_df=pd.read_csv('train_150k.csv','r',delimiter=',')
#data_df=data_df.drop(['id'],1)
data_df.info()

features = ['hour','C1','banner_pos','device_type','device_conn_type',
            'C14','C15','C16','C17','C18','C19','C20','C21','site_id','site_domain',
            'site_category','app_id', 'app_domain','app_category','device_model',
            'device_id','device_ip']

le = LabelEncoder()
for col in ['site_id','site_domain','site_category','app_id','app_domain',
            'app_category','device_model','device_id','device_ip']:
    le.fit(list(data_df[col]))
    data_df[col] = le.transform(data_df[col])
    
data_df.hour=data_df.hour.astype('str')
data_df['hour']=data_df.hour.str[6:8]
data_df.hour=data_df.hour.astype('int')

scaler = StandardScaler()
for col in features:
    scaler.fit(list(data_df[col]))
    data_df[col] = scaler.transform(data_df[col])
    
# Remove outliner
for col in features:
    # keep only the ones that are within +3 to -3 standard deviations in the column col,
    data_df = data_df[np.abs(data_df[col]-data_df[col].mean())<=(3*data_df[col].std())]
    # or if you prefer the other way around
    data_df = data_df[~(np.abs(data_df[col]-data_df[col].mean())>(3*data_df[col].std()))]    
          
import sklearn
print(sklearn.__version__)
x=data_df.drop(['click'],1)
y=data_df['click']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)

#Recursive feature elimination to select the optimal number of features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
regressor = RandomForestClassifier(n_estimators=200)
rfecv = RFECV(estimator=regressor, step=1, cv=10)
rfecv.fit(x_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new dataframe with features and their rankings
ranking=pd.DataFrame({'Features': x_train.columns})
ranking['rank'] = rfecv.ranking_
ranking.sort_values('rank',inplace=True)
ranking.to_csv('Ranking1.csv',index=False)

new_xtrain=x_train[['hour','banner_pos','device_type','device_conn_type',
            'C14','C15','C17','C18','C19','C20','C21','site_id','site_domain',
            'site_category','app_id', 'app_domain','app_category','device_model',
            'device_id','device_ip']]
new_ytrain=y_train
new_xtest=x_test[['hour','banner_pos','device_type','device_conn_type',
            'C14','C15','C17','C18','C19','C20','C21','site_id','site_domain',
            'site_category','app_id', 'app_domain','app_category','device_model',
            'device_id','device_ip']]
new_ytest=y_test

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


names = ["Extra Trees", "Random Forest", "KNeighbors","Logistic",
         "Naive Bayes", "Decision Tree","Support Vector Machine"]
classifiers = [
    ExtraTreesClassifier(n_estimators=200,criterion = 'entropy'),
    RandomForestClassifier(n_estimators=200,criterion = 'entropy'),
    KNeighborsClassifier(),
    LogisticRegression(),
    GaussianNB(),
    DecisionTreeClassifier(criterion='entropy'),
    SVC(kernel = 'rbf')
]

i=0
f1_results=[]
acc_results=[]
for classifier in classifiers:
    print(names[i])
    classifier.fit(new_xtrain, new_ytrain)
    y_pred = classifier.predict(new_xtest)
    f1score=f1_score(new_ytest,y_pred)
    accuracy=accuracy_score(new_ytest,y_pred)
    print("F1 Score:",f1score)
    print("Accuracy Score:",accuracy)
    f1_results.append(f1score)
    acc_results.append(accuracy)
    i+=1