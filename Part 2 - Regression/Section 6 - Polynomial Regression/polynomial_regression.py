#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:01:23 2018

@author: rxda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting Linear Regression to the datasets
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the datasets
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Linear Regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='green')
plt.title('Linear Regression')
plt.xlabel('Postion Level')
plt.ylabel('Salary')

# Polynomial Regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='green')
plt.title('Linear Regression')
plt.xlabel('Postion Level')
plt.ylabel('Salary')

#
x_grid= np.arange(min(x), max(x), 0.001)
x_grid= x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='green')
plt.title('Linear Regression')
plt.xlabel('Postion Level')
plt.ylabel('Salary')




