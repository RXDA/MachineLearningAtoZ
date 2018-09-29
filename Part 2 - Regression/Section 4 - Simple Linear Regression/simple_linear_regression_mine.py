# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:21:36 2018

@author: Admin
"""

# Simple Liner Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\\RXDA\\MachineLearningAtoZ\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv")
x=dataset.iloc[:, 0].values.reshape(-1, 1)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 1)

# Fitting Simple Liner Regression to the Traing Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the Test set results
y_pred=regressor.predict(x_test)

# Visualising the Training set results
# 画点
plt.scatter(x_train,y_train,color='red')
# 画线
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.show()


# Visualising the Test set results
# 画点
plt.scatter(x_train,y_train,color='red')
# 画线
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.show()