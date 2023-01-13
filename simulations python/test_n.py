import numpy as np

var_x = 1
var_y = 1
corr = 0.8
covariance = np.matrix([[var_x, corr * np.sqrt(var_x * var_y)], [corr * np.sqrt(var_x * var_y), var_y]])
# Compute the Decomposition:
A = np.linalg.cholesky(covariance)
mean = np.matrix([[0.], [1.]])
X= np.zeros((2,10))
Z = np.random.normal(size=(2, 1))
x= np.dot(A,Z) + mean
print(X)
print(x)

print(np.shape(x))
print(np.shape(X))





#replace the 2 row of X with 
X[0,1] = x[0]
X[1,1] = x[1]

print(X)






