import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


df = pd.read_csv("C:\\AllDesktop\\Prog\\python\\MANAS\\DataSheets\\new.csv")
print(df.head())

# now we divide the dataset into the target variable and the independent variable

#we will assign the independent variables as input

input = df.drop('salary_more_than_100k', axis='columns')
target = df['salary_more_than_100k']

# now, we all know that machine learning algorithm works on numbers only, not on labels or words.
# so we gotta convert the input columns into numbers!
# we use the label encoder from sklearn
# we are gonna create 3 different objects for three different columns

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# now in our input data frame, we create more columns

input['company_n'] = le_company.fit_transform(input['company'])
input['job_n'] = le_company.fit_transform(input['job'])
input['degree_n'] = le_company.fit_transform(input['degree'])
# replaces the labels with their corresponding numbers starting from 0 in alphabetical order!

# now we create a new daatframe with only the numbered columns

input_n = input.drop(['company', 'job', 'degree'], axis = 'comlumns')

model = tree.DecisionTreeClassifier()

model.fit(input_n, target)

model.score(input_n, target) # now here we will be getting a high score as the data is very simple

model.predict([[1, 1, 0]]) # in here we usually supply a dataframe here in 2D array form but we will supply the raw values from our input_n dataframe
