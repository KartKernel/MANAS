import matplotlib.pyplot as pp
import pandas as pd
import numpy as np

df = pd.read_csv('C:\\AllDesktop\\Prog\\python\\MANAS\\DataSheets\\single_LR.csv')
x = df.input.to_numpy()
y = df.output.to_numpy()

m = 0
c = 0
learning_rate = 0.0095
n = len(x)
epochs = 1000

for i in range(epochs):

    y_pred = m * x + c # scalar multiplication + matrix/vector addition --> y_pred is also an array

    # in first iteration, y_pred is a null vector as m and c is 0
    
    # sum() finds the sum of the elements of the array
    
    D_m = (-2/n) * sum(x * (y - y_pred)) 
    D_c = (-2/n) * sum(y - y_pred) 

    #update the parameters

    m = m - learning_rate * D_m
    c = c - learning_rate * D_c
    # so we update the value of the parameters every iteration and feed it to y_pred

# now finally with the optimal values of the parameters,

y_pred = m * x + c

print(y_pred)
print(x)
print(y)

pp.scatter(x, y)
pp.plot(x, y_pred)
pp.show()

