# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')
x = data.iloc[:,3:5].values #3，4lie

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):  #1-10
    kmeans = KMeans(n_clusters = i, max_iter = 300, n_init =10, init = 'k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method') #手肘方法
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, max_iter = 300, n_init =10, init = 'k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0,1], s = 100, c = 'red', label = 'c0')
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1,1], s = 100, c = 'blue', label = 'c1')
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2,1], s = 100, c = 'green', label = 'c2')
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3,1], s = 100, c = 'yellow', label = 'c3')
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4,1], s = 100, c = 'pink', label = 'c4')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c = 'black')
plt.title('Clusters of clients')
plt.show()
