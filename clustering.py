#-------------------------------------------------------------------------
# AUTHOR: Joshua Reyes
# FILENAME: clustering.py
# SPECIFICATION: using k-means to get a cluster on a graph
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64]
kshadows = []
high = 0
for k in range(2,21):
     kmeans = KMeans(n_clusters=k, random_state=0, n_init = 10)
     kmeans.fit(X_training)


     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     kshadows.append(silhouette_score(X_training, kmeans.labels_))
kmeans = KMeans(n_clusters=(kshadows.index(max(kshadows)) + 2), random_state=0, n_init = 10)
kmeans.fit(X_training) 
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(range(2,21), kshadows)
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df2 = pd.read_csv('testing_data.csv', header=None)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df2.values).reshape(1,len(df2))[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
