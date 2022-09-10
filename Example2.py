import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

data_file = pd.read_csv('C:\\AllDesktop\\Prog\\python\\MANAS\\DataSheets\\new.csv')

x = data_file.x.to_numpy()
y = data_file.y.to_numpy()

x = x/500
y = y/500

# initiating the vlue of m and c to 0

m = 0
c = 0

learning_rate = 0.7

epochs = 1000 # no of iterations 
# if we choose a smaller value for learning rate, we need a greater value of epochs

n = (len(x)) # no of inputs/ training examples
for i in range(epochs):
    # first we predict the value of y with the current value of m and c
    y_pred = m * x + c
    # finding the partial derivatives 
    D_m = (-2/n) * sum(x * (y - y_pred))
    D_c = (-2/n) * sum(y - y_pred)
    # updating the values of m and c
    m = m - (learning_rate*D_m)
    c = c - (learning_rate*D_c)

y_pred = m * x + c
pp.scatter(x, y)
pp.plot(x, y_pred)
pp.show()