# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:46:45 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

# 虚拟变量陷阱
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting Multiple Liner Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x_train = np.append(arr=np.ones((40,1)), values = x_train, axis=1)

x_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train ,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train ,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train ,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train ,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train ,exog = x_opt).fit()
regressor_OLS.summary()






