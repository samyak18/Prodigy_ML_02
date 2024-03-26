#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
np.random.seed(123)
n_customers = 200
n_products = 5
purchase_history = np.random.randint(0, 10, size=(n_customers, n_products))
columns = [f'Product_{i+1}' for i in range(n_products)]
data = pd.DataFrame(purchase_history, columns=columns)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
k = 6
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_data)
data['Cluster'] = cluster_labels
print(data.head())


# In[ ]:




