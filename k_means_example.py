from matplotlib import scale
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as pp

df = pd.read_csv('C:\AllDesktop\Prog\python\MANAS\DataSheets\income.csv')


pp.scatter(df['Age'], df['Income($)'])
pp.show()
# if there are many features, we can use PCA and reduce the number of features and then do this
"""
km = KMeans(n_clusters=3)

y_predicted = km.fit_predict(df[['Age', 'Income($)']]) 

# this statement ran the kmeans algorithm on age and income and computed the cluster as per our criteria where we told the algorithm to identify
# three clusters somehow

print(y_predicted) # so we have three clusters 0 1 and 2

df['Cluster'] = y_predicted

df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]

pp.scatter(df1.Age, df1['Income($)'], color='green')
pp.scatter(df2.Age, df2['Income($)'], color='red')
pp.scatter(df3.Age, df3['Income($)'], color='blue')
# we are gonna get a problem in the clusters as the income column is not scaled!
# when we dont scale our features properly, we might run into this problem


pp.xlabel('AGE')
pp.ylabel('Income')
pp.show()
"""

# scaling

scaler = MinMaxScaler()
df['Income($)'] = scaler.fit_transform(df[['Income($)', 'Age']])

df.Age = scaler.fit_transform(df[['Age', 'Income($)']])
print(df)

# now we have scaled income and age

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

df['Cluster'] = y_predicted
print(df.head())

df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]

pp.scatter(df1.Age, df1['Income($)'], color='green', label = 'incomegrp1')
pp.scatter(df2.Age, df2['Income($)'], color='red', label = 'incomegrp2')
pp.scatter(df3.Age, df3['Income($)'], color='blue', label = 'incomegrp3')
pp.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple', marker = '*', label = 'centroid')

pp.xlabel('AGE')
pp.ylabel('Income')
pp.legend()
pp.show()

# now we can see that the clustering is more accurate and there is no error

# using the elbow plot method as in real life problems, we will have a lot of features

k_rng = range(1, 10)
sse = [] # we will find sse for k = 1, then for k = 2 and keep storing it in sse

for i in k_rng:
    km1 = KMeans(n_clusters=i) # for each iteration, we make a new model with cluster = k
    km1.fit(df[['Age', 'Income($)']]) # we try to fit the dataframe 
    sse.append(km1.inertia_) # we append sse for each iteration to the sse array. inertia is a parameter that gives the sum of square error

pp.xlabel('K')
pp.ylabel('Sum of squared error')
pp.plot(k_rng, sse)
pp.show()
