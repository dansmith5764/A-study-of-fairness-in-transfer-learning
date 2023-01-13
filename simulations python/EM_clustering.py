import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


# create first guassian
d = 2 # Number of dimensions
mean = np.matrix([[0.], [1.]])

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
#X = np.concatenate((X, np.zeros((1, n))), axis=0)
#add a 1 to the end of each sample
#Y = np.concatenate((Y, np.ones((1, n))), axis=0)
 

def Gaussain_pdf(x1, x2, var_x, var_y, corr, mean): 
   
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
  

def density_function(x, y):
    # return 1/(pi**2*(1 + x**2)*(1 + y**2))
    return 1/(np.pi**2*np.multiply((1 + np.square(x)),(1 + np.square(y))))
    

# now combine the two datasets
Z = np.concatenate((X, Y), axis=1)
print('shape z', np.shape(Z))





# em clustering
from sklearn import mixture
# Fit a Gaussian mixture with EM using ten components
gmm = mixture.GaussianMixture(n_components=2, covariance_type="full", max_iter=100).fit(np.asarray(Z))


#print first 10 line of z
print(Z[:,0:10])
print ( 'predict ', gmm.predict(np.asarray(Z))) 
print ( 'means size  ', np.shape(gmm.means_))
print ( 'covariances size ', np.shape(gmm.covariances_))


#first 10 lines of z
print('z colum 1',Z[0,0:10])
print('z colum 2',Z[1,0:10])

# first 10 lines of mean 
print ( 'means    ', gmm.means_[0, 1:10])
print ( 'means  points ', gmm.means_[1, 1:10 ])

# # first 10 lines of covariance
# print ( 'covariances at column  1 ', gmm.covariances_[0, : ])

# #split into two datasets
# z1 = Z[:, gmm.means_ == 0]
# Z2 = Z[:, gmm.predict(np.asarray(Z)) == 1]

# print()





# # now we need to find the mean and vraiances of each cluster

# def find_mean(cluster):
#     #find mean of each cluster
#     mean_k_1 = int(abs(np.mean(cluster[0,:], axis=1)))
#     mean_k_2 = int(abs(np.mean(cluster[1,:], axis=1)))
#     mean_K_1 = np.matrix([[mean_k_1], [mean_k_2]])
#     return mean_K_1

# def find_covariance(cluster):
#     #find covariance of each cluster
#     #find correlation
#     var_k_1 = np.var(cluster, axis=1)
 
#     corr_k_1 = np.corrcoef(cluster[0,:], cluster[1,:])
#     corr_k_1 = corr_k_1[0,1]
#     return var_k_1,  corr_k_1

# # find mean and covariance of each cluster
# mean_K_1 = find_mean(cluster1)
# mean_K_2 = find_mean(cluster2)
# var_k_1, corr_k_1 = find_covariance(cluster1)
# var_k_2, corr_k_1_2 = find_covariance(cluster2)






