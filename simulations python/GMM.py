import numpy as np
import matplotlib.pyplot as plt


# create first guassian
d = 2 # Number of dimensions
mean = np.matrix([[0], [1]])

print('mean', mean)
var_x = 1
var_y = 1
corr = 0.8
covariance = np.matrix([[var_x, corr * np.sqrt(var_x * var_y)], [corr * np.sqrt(var_x * var_y), var_y]])

# create second guassian
mean2 = np.matrix([[5.], [6.]])
var_x2 = 1
var_y2 = 1
corr2 = 0.8
covariance2 = np.matrix([[var_x2, corr2 * np.sqrt(var_x2 * var_y2)], [corr2 * np.sqrt(var_x2 * var_y2), var_y2]])

 
# Compute the Decomposition:
A = np.linalg.cholesky(covariance)
A2 = np.linalg.cholesky(covariance2)
 
# Sample X from standard normal
n = 1000 # Samples to draw
Z = np.random.normal(size=(d, n))
Z1 = np.random.normal(size=(d, n))
 
# Apply the transformation
X = A.dot(Z) + mean
Y = A2.dot(Z1) + mean2
#add a 0 to the end of each sample
X = np.concatenate((X, np.zeros((1, n))), axis=0)
#add a 1 to the end of each sample
Y = np.concatenate((Y, np.ones((1, n))), axis=0)
 

def Gaussain_pdf(X, var_x, var_y, corr, mean): 
    x1 = X[0]
    x2 = X[1]
    var_x = var_x
    var_y = var_y
    corr = corr
    mean = mean

    a = 1 / (2 * np.pi *np.sqrt(var_x)*np.sqrt(var_y) * np.sqrt(1-corr**2))
    b = np.square(x1-mean[0]/var_x)
    c = np.square(x2-mean[1]/var_y)
    d = 2*corr*np.multiply((x1-mean[0]),(x2-mean[1]))/(np.sqrt(var_x)*np.sqrt(var_y))
    
   
    e = np.exp(-1/(2*(1-corr**2))*(b+c-d))
    return a*e
  

#3d scatter plot with labels 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# print all values going into gaussain pdf
print('X',X)
print('var_x',var_x)
print('var_y',var_y)
print('corr',corr)
print('mean',mean)

ax.scatter(X[0,:], X[1,:], Gaussain_pdf(X, var_x, var_y, corr, mean ), c='r', marker='o')
ax.scatter(Y[0,:], Y[1,:], Gaussain_pdf(Y, var_x2, var_y2, corr2, mean2 ), c='b', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('pdf')
plt.show()





