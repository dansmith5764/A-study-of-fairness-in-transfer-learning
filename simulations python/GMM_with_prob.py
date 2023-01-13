import numpy as np
import matplotlib.pyplot as plt




# Sample X from standard normal
n = 1000 # Samples to draw

def N_mulitvarate_guassian(d, mean, mean2, n, prob, covariance):
   
 
    # Compute the Decomposition:
    A = np.linalg.cholesky(covariance)
    A2 = np.linalg.cholesky(covariance2)

 

  
    X = np.zeros((d+1,n))
    print ('shape ', np.shape(X))
    for i in range(n):
        #prob draw from first guassian
    
        if np.random.rand() < prob:
            Z = np.random.normal(size=(d, 1))
            #shape X
            v = np.dot(A,Z) + mean
            X[0:3,i-1:i]= v
            # add the label
            X[3,i-1:i] = 0

        else:
        #prob draw from second guassian
            Z = np.random.normal(size=(d, 1))
            v= np.dot(A2,Z) + mean2
            X[0:3,i-1:i]= v
            # add the label
            X[3,i-1:i] = 1

            
    return X

 
# Apply the transformation at set probility
def bivarate_guassian(mean, mean2, n, prob, covariance, covariance2):
   
    var_x = covariance[0,0]
    var_y = covariance[1,1]
    #corr = covariance[0,1]/np.sqrt(var_x*var_y)
    


    # Compute the Decomposition:
    A = np.linalg.cholesky(covariance)
    #A2 = np.linalg.cholesky(covariance2)

  
    X = np.zeros((d+1,n))
    print ('shape ', np.shape(X))
    for i in range(n):
        #prob draw from first guassian
        if np.random.rand() < prob:
            Z = np.random.normal(size=(d, 1))
            X[0,i]= np.dot(A,Z) + mean[0]

        else:
        #prob draw from second guassian
            Z = np.random.normal(size=(d, 1))
            X[1,i]= np.dot(A,Z) + mean[1]
            #replace the i-th row of X with x
            
            #X = np.concatenate((X, np.ones((1, n))), axis=0)
    return X



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


def KLDivergence(P, Q):
    result = np.sum(np.multiply(P, np.log(np.divide(P, Q))))
    return result


# parameter for first guassian
d = 2 # Number of dimensions
prob = 0.1
mean = np.matrix([[0.], [1.]])


var_G1 = 1
var_G2 = 1
corr = 0.8

# parameters for second guassian
mean2 = np.matrix([[2.], [2.]])
covariance = np.matrix([[1, 0.8], [0.8, 1]])
covariance2 = np.matrix([[1, 0.8], [0.8, 1]])

mean3 = np.matrix([[4.], [4.]])

X = bivarate_guassian(mean, mean2,n, prob, var_G1, var_G2, corr)
Y = bivarate_guassian(mean, mean2,n, prob, var_x2, var_y2, corr)

# Compute the KL divergence
P = Gaussain_pdf(X, var_x, var_y, corr, mean)
Q = Gaussain_pdf(Y, var_x2, var_x, corr2, mean2)

#plot the data
# plot X1
plt.scatter(X[0,:], X[1,:],  c='r', marker='o')
# plot X2
plt.scatter(Y[0,:], Y[1,:],  c='b', marker='o')
plt.show()




#size p
print ('size p', np.shape(P))
print ('size q', np.shape(Q))

P = P[0,0:999]
Q = Q[0,0:999]

# print('sum of log(p/q)',np.sum(np.log(np.divide(P[0,0:999], Q[0,0:999]))))
# print('sum of P',np.sum(P[0,0:999]))
# result = np.sum(np.multiply(P, np.log(np.divide(P[0,999], Q[0,999]))))
# print('result', result)
print('KL divergence', KLDivergence(P, Q))
