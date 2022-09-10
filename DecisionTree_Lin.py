import pandas as pd
from sklearn import tree
import matplotlib.pyplot as pp
import sklearn

df = pd.read_csv("C:\AllDesktop\Prog\python\MANAS\DataSheets\linear_regression_dataset.csv")

temp_x = df.drop('TOTCHG', axis = 'columns')
temp_y = df.TOTCHG

x_train = temp_x.iloc[0:20, :]
y_train = temp_y.iloc[0:20]

x_test = temp_x.iloc[20:40, :]
y_test = temp_y.iloc[20:40]

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)

print(model.predict(x_test))

sklearn.tree.plot_tree(model.fit(x_train, y_train), filled = True)
pp.figure(figsize= (210, 210))
pp.show()