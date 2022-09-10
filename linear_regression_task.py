import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

data_file = pd.read_csv('C:\AllDesktop\Prog\python\MANAS\DataSheets\linear_regression_dataset.csv')

x = data_file.iloc[0:, 3:5]
x['x0'] = 1
x = np.array(x)

y = data_file.iloc[0:, 5:]
y = np.array(y)

n = len(x)

def cost(x, params):
    total_cost = 0
    for i in range(n):
        total_cost += (1/n) * ((x[i] * params).sum() - y[i])**2
    return total_cost
    

def gradient_descent(x, params, learning_rate, epochs):
    tcost = []
    for i in range(epochs):
        slopes = np.zeros(3)
        for j in range(n):
            for k in range(3):
                slopes[k] += (1/n) * ((x[j] * params).sum() - y[j]) * x[j][k]

        params = params - learning_rate * slopes

        if epochs % 100 == 0:
            tcost.append(cost(x, params))
         

    return params, tcost




params = np.zeros(3)
learning_rate = 0.000001
epochs = 1000
params, tcost = gradient_descent(x, params, learning_rate, epochs)
#tcost = cost(x, params)
print("the parameters of the hypothesis are : " + str(params))
pp.plot(tcost)
pp.plot()
pp.xlabel('iterations')
pp.ylabel('TOTCHG')
pp.show()