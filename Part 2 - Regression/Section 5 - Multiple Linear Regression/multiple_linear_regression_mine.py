# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:46:45 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("D:\\RXDA\\MachineLearningAtoZ\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

# 虚拟变量陷阱
# x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting Multiple Liner Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)




