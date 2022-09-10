import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

data = pd.read_csv('C:\AllDesktop\Prog\python\MANAS\DataSheets\multiple_LR.csv')

# all the features in our x arrayprint(data)

x1 = data.area.to_numpy()
x2 = data.bedrooms.to_numpy()
y = data.price.to_numpy()

x = data.iloc[0:, 0:2] # iloc takes all the training example in an array from [a:b] meaning from a to b exclusive of b inclusive of a
                       # all the rows and all the columns except price is in x as array

# now we need to add an extra feature by ourselves which is x0 = 1

x['x0'] = 1

# now we convert x to a numpy array for easier calculations

x = np.array(x)

# now we convert prices to a numpy array y

y = data.iloc[0:, 2:]
y = np.array(y)
n = len(x)

# so now we have all the features in x and the price in y as array


# gradient descent function

def gradient_descent(x, params, learning_rate, epochs):
    for i in range(epochs):
        slopes = np.zeros(3)
        for j in range(n): # in each iteration we will be passing our data once so the j loop with range = no of target values
            for k in range(3): # in each training examples (row), we are going to be passing through all features once
                slopes[k] += (1/n) * ((x[j] * params).sum() - y[j]) * x[j][k] # (data[j] * params).sum() is the hypothesis for that row
                                                                         #kth feature in the jth row
                # we are using sum() for hypothesis because it will keep adding theta(k) * x(k) to it 
        # after obtaining slopes, at the end of each epoch, we can update our parameters
        params = params - learning_rate * slopes
    
    return params
    

# running the gradient descent


params = np.zeros(3) # initiating the parameters as 0 (theta0 theta1 theta2, ...)
learning_rate = 0.001
epochs = 1000
params = gradient_descent(x, params, learning_rate, epochs)
print(params)



"""
ax = pp.axes(projection = '3d')
pp.plot()
pp.show()
"""
