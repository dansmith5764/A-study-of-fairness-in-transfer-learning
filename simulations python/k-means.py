import numpy as np
import matplotlib.pyplot as plt


# parameter for first guassian
d = 2 # Number of dimensions
prob = 0.1
mean = np.matrix([[0.], [1.]])

print('mean', mean)
var_x = 1
var_y = 1
corr = 0.8
covariance = np.matrix([[var_x, corr * np.sqrt(var_x * var_y)], [corr * np.sqrt(var_x * var_y), var_y]])

# parameters for second guassian
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


Z = np.random.normal(size=(d, 1))
x = np.dot(A,Z) + mean
# print(x)
# print (np.shape(x))
 
# Apply the transformation at set probility
def sample_guassian(mean,n, prob):
    X = np.zeros((d+1,n))
    print ('shape ', np.shape(X))
    for i in range(n):
        if np.random.rand() < prob:
            Z = np.random.normal(size=(d, 1))
            x= np.dot(A,Z) + mean
            #replace the i-th row of X with x
            X[0,i] = x[0]
            X[1,i] = x[1]
            X[2,i] = 0

            
            #add a 0 to the end of each sample
            #X = np.concatenate((X, np.zeros((1, n))), axis=0)
        else:
            Z = np.random.normal(size=(d, 1))
            x= np.dot(A2,Z) + mean2
            #replace the i-th row of X with x
            X[0,i] = x[0]
            X[1,i] = x[1]
            X[2,i] = 1
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

# # def density_function(x, y):
# #     # return 1/(pi**2*(1 + x**2)*(1 + y**2))
# #     return 1/(np.pi**2*np.multiply((1 + np.square(x)),(1 + np.square(y))))
    
#3d scatter plot with labels 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = sample_guassian(mean,  n, prob)

print ('shape x', np.shape(X))



#split the data into two classes
X1 = X[:,   X[2,:] == 0]
X2 = X[:,   X[2,:] == 1]

print ('shape x1', np.shape(X1))
print ('shape x2', np.shape(X2))


# # plot X1
# ax.scatter(X1[0,:], X1[1,:],  c='r', marker='o')
# # plot X2
# ax.scatter(X2[0,:], X2[1,:],  c='b', marker='o')
# plt.show()

ax.scatter(X1[0,:], X1[1,:], Gaussain_pdf(X1, var_x, var_y, corr, mean), c='r', marker='o')
ax.scatter(X2[0,:], X2[1,:], Gaussain_pdf(X2, var_x2, var_y2, corr2, mean2 ), c='b', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('pdf')
plt.show()

# # # # now combine the two datasets
# # # Z = np.concatenate((X, Y), axis=1)
# # # print(np.shape(Z))


# # k-means clustering
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=2, random_state=0).fit(np.asarray(X))
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)
# print(kmeans.n_clusters)

# # number of samples in each cluster
# print(np.bincount(kmeans.labels_))

# # seprate the two clusters
# cluster1 = X[:,kmeans.labels_==0]
# cluster2 = X[:,kmeans.labels_==1]


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



# from scipy.stats import norm
# # sub  3d plot of the two clusters and the pdfs
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')

# #plot the original data
# ax.scatter(X1[0,:], X1[1,:], Gaussain_pdf(X1, var_x, var_y, corr, mean ), c='r', marker='o')
# ax.scatter(X2[0,:], X2[1,:], Gaussain_pdf(X2, var_x2, var_y2, corr2, mean2 ), c='b', marker='o')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('pdf')
# ax.set_title('Original Data')

# # #plot the clusters
# # ax = fig.add_subplot(1, 2, 2, projection='3d')


# # ax.scatter(cluster1[0,:], cluster1[1,:],  Gaussain_pdf(cluster1, var_k_1[0], var_k_1[1], corr_k_1, mean_K_1 ),  c='r', marker='o')
# # ax.scatter(cluster2[0,:], cluster2[1,:],  Gaussain_pdf(cluster2, var_k_2[0], var_k_2[1], corr_k_1_2, mean_K_2 ),  c='b', marker='o')
# # ax.set_xlabel('x1')
# # ax.set_ylabel('x2')
# # ax.set_zlabel('pdf')
# # ax.set_title('K-meClustered Data')
# plt.show()





