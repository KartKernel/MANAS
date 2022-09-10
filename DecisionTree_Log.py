import imp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("C:\AllDesktop\Prog\python\MANAS\DataSheets\LogRegression_weatherAUS.csv")

df.MinTemp.fillna(df.MinTemp.median(skipna = True), inplace = True)
df.MaxTemp.fillna(df.MaxTemp.median(skipna = True), inplace = True)

df.Rainfall.fillna(df.Rainfall.mean(skipna = True), inplace = True)

df.Evaporation.fillna(df.Evaporation.median(skipna = True), inplace = True)

df.Sunshine.fillna(df.Sunshine.median(skipna = True), inplace = True)

df.WindGustSpeed.fillna(df.WindGustSpeed.mean(skipna = True), inplace = True)

df.WindSpeed9am.fillna(df.WindSpeed9am.median(skipna = True), inplace = True)
df.WindSpeed3pm.fillna(df.WindSpeed3pm.median(skipna = True), inplace = True)

df.Humidity9am.fillna(df.Humidity9am.median(skipna = True), inplace = True)
df.Humidity3pm.fillna(df.Humidity3pm.median(skipna = True), inplace = True)

df.Pressure9am.fillna(df.Pressure9am.median(skipna = True), inplace = True)
df.Pressure3pm.fillna(df.Pressure3pm.median(skipna = True), inplace = True)

df.Cloud9am.fillna(df.Cloud9am.median(skipna = True), inplace = True)
df.Cloud3pm.fillna(df.Cloud3pm.median(skipna = True), inplace = True)

df.Temp9am.fillna(df.Temp9am.median(skipna = True), inplace = True)
df.Temp3pm.fillna(df.Temp3pm.median(skipna = True), inplace = True)

sns.countplot(x = 'WindGustDir', data = df)

# most occuring is W

df.WindGustDir.fillna("W", inplace = True)


sns.countplot(x = 'WindDir9am', data = df)

# most occuring is N

df.WindDir9am.fillna("N", inplace = True)

sns.countplot(x = 'WindDir3pm', data = df)

# most occuring is SE

df.WindDir3pm.fillna("SE", inplace = True)

sns.countplot(x = 'RainToday', data = df)

# most occuring is No

df.RainToday.fillna("No", inplace = True)

sns.countplot(x = 'RainTomorrow', data = df)

# most occuring is No

df.RainTomorrow.fillna("No", inplace = True)


le_WindGustDir = LabelEncoder()
le_WindDir9am = LabelEncoder()
le_WindDir3pm = LabelEncoder()

le_RainToday = LabelEncoder()
le_RainTomorrow = LabelEncoder()

df['WindGustDir'] = le_WindGustDir.fit_transform(df.WindGustDir)
df['WindDir9am'] = le_WindDir9am.fit_transform(df.WindDir9am)
df['WindDir3pm'] = le_WindDir3pm.fit_transform(df.WindDir3pm)

df['RainToday'] = le_RainToday.fit_transform(df.RainToday)
df['RainTomorrow'] = le_RainTomorrow.fit_transform(df.RainTomorrow)



scaler = MinMaxScaler()

df.MaxTemp = scaler.fit_transform(df[['MaxTemp', 'MinTemp']])
df.MinTemp = scaler.fit_transform(df[['MinTemp', 'MaxTemp']])

df.Sunshine = scaler.fit_transform(df[['Sunshine', 'Evaporation']])
df.Evaporation = scaler.fit_transform(df[['Evaporation', 'Sunshine']])

df.WindGustDir = scaler.fit_transform(df[['WindGustDir', 'Sunshine']])

df.WindGustSpeed = scaler.fit_transform(df[['WindGustSpeed', 'WindGustDir']])

df.WindDir9am = scaler.fit_transform(df[['WindDir9am', 'WindDir3pm']])
df.WindDir3pm = scaler.fit_transform(df[['WindDir3pm', 'WindDir9am']])

df.WindSpeed9am = scaler.fit_transform(df[['WindSpeed9am', 'WindSpeed3pm']])
df.WindSpeed3pm = scaler.fit_transform(df[['WindSpeed3pm', 'WindSpeed9am']])

df.Humidity9am = scaler.fit_transform(df[['Humidity9am', 'Humidity3pm']])
df.Humidity3pm = scaler.fit_transform(df[['Humidity3pm', 'Humidity9am']])

df.Pressure9am = scaler.fit_transform(df[['Pressure9am', 'Pressure3pm']])
df.Pressure3pm = scaler.fit_transform(df[['Pressure3pm', 'Pressure9am']])

df.Cloud9am = scaler.fit_transform(df[['Cloud9am', 'Cloud3pm']])
df.Cloud3pm = scaler.fit_transform(df[['Cloud3pm', 'Cloud9am']])

df.Temp9am = scaler.fit_transform(df[['Temp9am', 'Temp3pm']])
df.Temp3pm = scaler.fit_transform(df[['Temp3pm', 'Temp9am']])



temp_x = df.drop('RainTomorrow', axis = 'columns')
temp_y = df.RainTomorrow

x_train = temp_x.iloc[:1000, 2:]

y_train = temp_y.iloc[0:1000]

x_test = temp_x.iloc[1000:2000, 2:]

y_test = temp_y.iloc[1000:2000]


model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
model.predict(x_test)

print("the accuracy of the model is : ", model.score(x_test, y_test))