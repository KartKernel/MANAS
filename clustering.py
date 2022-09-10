from matplotlib import scale
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as pp


# i ll use either minmaxscaler or standardscaler
# StandardScaler does not guarantee balanced feature scales, 
# due to the influence of the outliers while computing the empirical mean and standard deviation.

df = pd.read_csv('C:\AllDesktop\Prog\python\MANAS\DataSheets\logistic_regression_weatherAUS.csv')

print(df.head())

# replacing the nan values

df.MinTemp.fillna(df.MinTemp.median(skipna=True), inplace = True)
df.MaxTemp.fillna(df.MaxTemp.median(skipna=True), inplace = True)

df.Rainfall.fillna(df.Rainfall.median(skipna=True), inplace = True)

df.Evaporation.fillna(df.Evaporation.median(skipna=True), inplace = True)

df.WindSpeed9am.fillna(df.WindSpeed9am.median(skipna=True), inplace = True)
df.WindSpeed3pm.fillna(df.WindSpeed3pm.median(skipna=True), inplace = True)

df.Humidity9am.fillna(df.Humidity9am.median(skipna=True), inplace = True)
df.Humidity3pm.fillna(df.Humidity3pm.median(skipna=True), inplace = True)

df.Pressure9am.fillna(df.Pressure9am.median(skipna=True), inplace = True)
df.Pressure3pm.fillna(df.Pressure3pm.median(skipna=True), inplace = True)

df.Cloud9am.fillna(df.Cloud9am.median(skipna=True), inplace = True)
df.Cloud3pm.fillna(df.Cloud3pm.median(skipna=True), inplace = True)

df.Temp9am.fillna(df.Temp9am.median(skipna=True), inplace = True)
df.Temp3pm.fillna(df.Temp3pm.median(skipna=True), inplace = True)

print(df.head())


# scaling the columns

scaler = MinMaxScaler()


df.Pressure9am = scaler.fit_transform(df[['Pressure9am', 'Pressure3pm']])
df.Pressure3pm = scaler.fit_transform(df[['Pressure3pm', 'Pressure9am']])

df.Temp9am = scaler.fit_transform(df[['Temp9am', 'Temp3pm']])
df.Temp3pm = scaler.fit_transform(df[['Temp3pm', 'Temp9am']])

df.Humidity9am = scaler.fit_transform(df[['Humidity9am', 'Humidity3pm']])
df.Humidity3pm = scaler.fit_transform(df[['Humidity3pm', 'Humidity9am']])

df.WindSpeed9am = scaler.fit_transform(df[['WindSpeed9am', 'WindSpeed3pm']])
df.WindSpeed3pm = scaler.fit_transform(df[['WindSpeed3pm', 'WindSpeed9am']])

df.MaxTemp = scaler.fit_transform(df[['MaxTemp', 'MinTemp']])
df.MinTemp = scaler.fit_transform(df[['MinTemp', 'MaxTemp']])

df.Rainfall = scaler.fit_transform(df[['Rainfall', 'Evaporation']])
df.Evaporation = scaler.fit_transform(df[['Evaporation', 'Rainfall']])

df.Cloud9am = scaler.fit_transform(df[['Cloud9am', 'Cloud3pm']])
df.Cloud3pm = scaler.fit_transform(df[['Cloud3pm', 'Cloud9am']])

print(df.head(10))

# to apply PCA, we are gonna allocate the df into a np array

x = df.iloc[0:100, 2:16]

print(x)

# converting to np array

x = x.values

print(x)

# applying pca 

pca = PCA(2)
X_PCA = pca.fit_transform(x)
print(X_PCA)

# X_PCA = np.absolute(X_PCA)

# converting the np array back to pandas data frame with the principal components 1 and 2

new_df = pd.DataFrame(X_PCA, columns = ['PC1', 'PC2'])
print(new_df.head(10))

# how it looks

pp.scatter(new_df.PC1, new_df.PC2)
pp.show()

# using the elbow method to find number of clusters

k_rng = range(1, 10)
sse = []

for i in k_rng:
    km = KMeans(n_clusters=i)
    km.fit(new_df[['PC1', 'PC2']])
    sse.append(km.inertia_)
    
pp.xlabel('K')
pp.ylabel('Sum of squared error')
pp.plot(k_rng, sse)
pp.show()

km1 = KMeans(n_clusters=3)
y_predicted = km1.fit_predict(new_df[['PC1', 'PC2']])

new_df['Cluster'] = y_predicted

print(new_df)

# dividing the daatframe into different clusters

from sklearn.cluster import KMeans

new_dfa = new_df[new_df.Cluster == 0] # new_dfa will contain all the entries that are in cluster 0 and so on
new_dfb = new_df[new_df.Cluster == 1]
new_dfc = new_df[new_df.Cluster == 2]

# visualising the clusters and the centroids

pp.scatter(new_dfa.PC1, new_dfa.PC2, color = 'green', label = 'cluster1')
pp.scatter(new_dfb.PC1, new_dfb.PC2, color = 'red', label = 'cluster2')
pp.scatter(new_dfc.PC1, new_dfc.PC2, color = 'blue', label = 'cluster3')
pp.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], color = 'purple', marker = '*', label = 'centroid')

pp.legend()
pp.show()