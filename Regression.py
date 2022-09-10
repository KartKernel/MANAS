import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import seaborn as sns

data_file = pd.read_csv('C:\AllDesktop\Prog\python\MANAS\DataSheets\logistic_regression_weatherAUS.csv')

# print(data_file.isnull().sum())

"""
bin=
It is a type of bar graph. To construct a histogram, the first step is to “bin” the range of values 
that is, divide the entire range of values into a series of intervals — and then count how many values fall into each interval"""

# MINIMUM TEMPERATURE

ax = data_file["MinTemp"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["MinTemp"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'MIN TEMPERATURE')
pp.xlim(-10, 30)
pp.show()


print("the mean temp is : ", data_file["MinTemp"].mean(skipna=True)) # 12.19
print("the median temperature is : ", data_file["MinTemp"].median(skipna="True")) # 12

# using median to replace the NA values in min temp

data_file["MinTemp"].fillna(data_file["MinTemp"].median(skipna=True), inplace=True)


# MAXIMUM TEMPERATURE


ax = data_file["MaxTemp"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["MaxTemp"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'MAX TEMPERATURE')
pp.xlim(-10, 60)
pp.show()


print("the mean max temp is : ", data_file["MaxTemp"].mean(skipna=True))
print("the median max temp is : ", data_file["MaxTemp"].median(skipna=True))

# using the median value to replace the NA values in max temp

data_file["MaxTemp"].fillna(data_file["MaxTemp"].median(skipna=True), inplace=True)

# RAINFALL

print("the mean rainfall is : ", data_file["Rainfall"].mean(skipna=True))
print("the median rainfall is : ", data_file["Rainfall"].median(skipna=True))

# using mean rainfall to replace the NA values of rainfall

data_file["Rainfall"].fillna(data_file["Rainfall"].mean(skipna=True), inplace=True)

# EVAPORATION


ax = data_file["Evaporation"].hist(bins=100, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Evaporation"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'EVAPORATION')
pp.xlim(-10, 60)
pp.show()


print("the mean evaporation is : ", data_file["Evaporation"].mean(skipna=True))
print("the median evaporation is : ", data_file["Evaporation"].median(skipna=True))

# using mean evaporation to replace the NA values of evaporation

data_file["Evaporation"].fillna(data_file["Evaporation"].mean(skipna=True), inplace=True)

# WINDSPEED 9AM


data_file["WindSpeed9am"].plot(kind = 'hist', color='teal')
pp.xlabel("WINDSPEED 9AM")
pp.xlim(-10, 60)
pp.show()


print("the mean windspeed at 9am is : ", data_file["WindSpeed9am"].mean(skipna=True))
print("the median windspeed at 9am is : ", data_file["WindSpeed9am"].median(skipna=True))

# using median windspeed at 9am ro replace the NA values

data_file["WindSpeed9am"].fillna(data_file["WindSpeed9am"].median(skipna=True), inplace=True)

# WINDSPEED 3PM


ax = data_file["WindSpeed3pm"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["WindSpeed3pm"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'WindSpeed3pm')
pp.xlim(-10, 65)
pp.show()


print("the mean windspeed at 3pm is : ", data_file["WindSpeed3pm"].mean(skipna=True))
print("the median windspeed at 3pm is : ", data_file["WindSpeed3pm"].median(skipna=True))

# using the median to replace the NA values in windspeed 3pm

data_file["WindSpeed3pm"].fillna(data_file["WindSpeed3pm"].median(skipna=True), inplace=True)

# HUMIDITY AT 9AM


ax = data_file["Humidity9am"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Humidity9am"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'Humidity9am')
pp.xlim(-5, 120)
pp.show()




print("the mean humidity at 9am is : ", data_file["Humidity9am"].mean(skipna=True))
print("the median humidity at 9am is : ", data_file["Humidity9am"].median(skipna=True))

# using the median to replace the NA values in humidity at 9am

data_file["Humidity9am"].fillna(data_file["Humidity9am"].median(skipna=True), inplace=True)

# HUMIDITY AT 3PM



ax = data_file["Humidity3pm"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Humidity3pm"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'Humidity3pm')
pp.xlim(-5, 120)
pp.show()

print("the mean humidity at 3pm is : ", data_file["Humidity3pm"].mean(skipna=True))
print("the median humidity at 3pm is : ", data_file["Humidity3pm"].median(skipna=True))

data_file["Humidity3pm"].fillna(data_file["Humidity3pm"].median(skipna=True), inplace=True)

# PRESSURE 9AM


ax = data_file["Pressure9am"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Pressure9am"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'Pressure9am')
pp.xlim(980, 1050)
pp.show()



print("the mean pressure at 9am is : ", data_file["Pressure9am"].mean(skipna=True))
print("the median pressure at 9am is : ", data_file["Pressure9am"].median(skipna=True))

# using median pressure at 9am to replace the NA values 

data_file["Pressure9am"].fillna(data_file["Pressure9am"].median(skipna=True), inplace=True)

# PRESSURE AT 3PM

# Pressure3pm

ax = data_file["Pressure3pm"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Pressure3pm"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'PRESSURE AT 3PM')
pp.xlim(980, 1050)
pp.show()

print("the mean pressure at 3pm is : ", data_file["Pressure3pm"].mean(skipna=True))
print("the median pressure at 3pm is : ", data_file["Pressure3pm"].median(skipna=True))

# using the median of pressure at 3pm to replace NA values

data_file["Pressure3pm"].fillna(data_file["Pressure3pm"].median(skipna=True), inplace=True)

# CLOUD AT 9AM

#Cloud9am


print("the cloud distribution at 9am is as shown below: ")
print(data_file['Cloud9am'].value_counts())
sns.countplot(x = 'Cloud9am', data=data_file, palette='Set2')
pp.xlabel("CLOUD AT 9AM")
pp.ylabel("Occurence")
pp.show()


print("the mean cloud at 9am is : ", data_file["Cloud9am"].mean(skipna=True))
print("the median cloud at 9am is : ", data_file["Cloud9am"].median(skipna=True))

data_file["Cloud9am"].fillna(data_file["Cloud9am"].median(skipna=True), inplace=True)

# CLOUD AT 3PM

# Cloud3pm


print("the cloud distribution at 3pm is as shown below: ")
print(data_file['Cloud3pm'].value_counts())
sns.countplot(x = 'Cloud3pm', data=data_file, palette='Set2')
pp.xlabel("CLOUD AT 3PM")
pp.ylabel("Occurence")
pp.show()

print("the mean cloud at 9am is : ", data_file["Cloud3pm"].mean(skipna=True))
print("the median cloud at 9am is : ", data_file["Cloud3pm"].median(skipna=True))

data_file["Cloud3pm"].fillna(data_file["Cloud3pm"].median(skipna=True), inplace=True)

# TEMPERATURE AT 9AM

# Temp9am


ax = data_file["Temp9am"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Temp9am"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'TEMPERATURE AT 9AM')
pp.xlim(-10, 40)
pp.show()

print("the mean temperature at 9am is : ", data_file["Temp9am"].mean(skipna=True))
print("the median temperature at 9am is : ", data_file["Temp9am"].median(skipna=True))

# using median temperature at 9am to replace the NA values 

data_file["Temp9am"].fillna(data_file["Temp9am"].median(skipna=True), inplace=True)

# TEMPERATURE AT 3PM

# Temp3pm


ax = data_file["Temp3pm"].hist(bins=30, density = True, stacked = True, color = 'teal', alpha = 0.4)
data_file["Temp3pm"].plot(kind = 'density', color='teal')
ax.set(xlabel = 'TEMPERATURE AT 3PM')
pp.xlim(-5, 50)
pp.show()



print("the mean temperature at 3pm is : ", data_file["Temp3pm"].mean(skipna=True))
print("the median temperature at 3pm is : ", data_file["Temp3pm"].median(skipna=True))

# using the median temperature at 3pm to replace the NA values

data_file["Temp3pm"].fillna(data_file["Temp3pm"].median(skipna=True), inplace=True)

# RAIN TOMORROW

# RainTomorrow


print("Rain occurs tomorrow is shown below: ")
print(data_file['RainTomorrow'].value_counts())
sns.countplot(x = 'RainTomorrow', data=data_file, palette='Set2')
pp.xlabel("RAIN TOMORROW")
pp.ylabel("Occurence")
pp.show()



# replacing NA values with 0 as rain not occuring is more frequent than rain occuring

data_file["RainTomorrow"].fillna(0, inplace=True)

# finished managing the datafile

# now we can start the logistic regression

data_file["Pressure9am"] = data_file["Pressure9am"]/100
data_file["Pressure3pm"] = data_file["Pressure3pm"]/100
data_file["WindSpeed3pm"] = data_file["WindSpeed3pm"]/10
data_file["WindSpeed9am"] = data_file["WindSpeed9am"]/10
data_file["Humidity9am"] = data_file["Humidity9am"]/10
data_file["Humidity9am"] = data_file["Humidity9am"]/10


x_train = data_file.iloc[0:1000, 2:16]
y_train = data_file.iloc[0:1000, 16:]

x_test = data_file.iloc[1000:2000, 2:16]
y_test = data_file.iloc[1000:2000, 16:]

# converting into np array
x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

n = x_train.shape[1] # no of features
m = x_train.shape[0] # no of observation



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_regression(x_train, y_train, epochs, learning_rate):

    cost_list = []
    
    W = np.zeros((n, 1)) # 14 x 1
    B = 0
    for i in range(epochs):

        Z = np.dot(x_train, W) + B
        A = sigmoid(Z) # A is our probablistic predictions ---> y_pred

        cost = -(1/m) * np.sum(y_train * np.log(A) + (1 - y_train) * np.log(1 - A))

        dW = (1/m) * np.dot(x_train.T, A - y_train)
        dB = (1/m) * np.sum(A - y_train)

        W = W - learning_rate * dW
        B = B - learning_rate * dB

        cost_list.append(cost)
        print("cost : ", cost)

    return W, B, cost_list 



epochs = 1000
learning_rate = 0.001
W, B, cost_list = log_regression(x_train, y_train, epochs, learning_rate)

pp.plot(cost_list)
pp.xlabel("epochs")
pp.ylabel("cost/error")
pp.show()


def accuracy(X, Y, W, B):
    
    Z = np.dot(X, W) + B
    A = sigmoid(Z)

    A = A > 0.5
    A = np.array(A, dtype='int64')

    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[0]) * 100

    print("the accuracy of the model is : ", acc, "%")

accuracy(x_test, y_test, W, B)



