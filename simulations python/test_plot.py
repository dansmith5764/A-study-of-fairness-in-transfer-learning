import numpy as np
import matplotlib.pyplot as plt


# parameter for first guassian
d = 3 # Number of dimensions
prob = 0.1
mean = np.matrix([[0.], [1.], [2.]])

print('mean', mean)
var_x = 1
var_y = 1
var_z = 1	
corr = 0.8
covariance = np.matrix([[var_x, corr * np.sqrt(var_x * var_y), corr * np.sqrt(var_x * var_z)], [corr * np.sqrt(var_x * var_y), var_y, corr * np.sqrt(var_y * var_z)], [corr * np.sqrt(var_x * var_z), corr * np.sqrt(var_y * var_z), var_z]])

# parameters for second guassian
mean2 = np.matrix([[5.], [6.], [7.]])
var_x2 = 1
var_y2 = 1
var_z2 = 1
corr2 = 0.8
covariance2 = np.matrix([[var_x2, corr2 * np.sqrt(var_x2 * var_y2), corr2 * np.sqrt(var_x2 * var_z2)], [corr2 * np.sqrt(var_x2 * var_y2), var_y2, corr2 * np.sqrt(var_y2 * var_z2)], [corr2 * np.sqrt(var_x2 * var_z2), corr2 * np.sqrt(var_y2 * var_z2), var_z2]])




 
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
def sample_guassian(mean, mean2, covariance, covariance2,  n, prob, d):
    # Compute the Decomposition:
    A = np.linalg.cholesky(covariance)
    A2 = np.linalg.cholesky(covariance2)

    print ('shape A ', np.shape(A))
 


    X = np.zeros((d+1,n))
    print ('shape X ', np.shape(X))
    for i in range(n):
        if np.random.rand() < prob:
            Z = np.random.normal(size=(d, 1))
            x= np.dot(A,Z) + mean

        
            #replace the i-th row of X with x
            X[0,i] = x[0]
            X[1,i] = x[1]
            X[2,i] = x[2]
            X[3,i] = 0

            
            #add a 0 to the end of each sample
            #X = np.concatenate((X, np.zeros((1, n))), axis=0)
        else:
            Z = np.random.normal(size=(d, 1))
            x= np.dot(A2,Z) + mean2
            #replace the i-th row of X with x
            X[0,i] = x[0]
            X[1,i] = x[1]
            X[2,i] = x[2]	
            X[3,i] = 1
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


#3d scatter plot with labels 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = sample_guassian(mean, mean2, covariance, covariance2,  n, prob, d)

print ('shape x', np.shape(X))



#split the data into two classes
X1 = X[:,   X[3,:] == 0]
X2 = X[:,   X[3,:] == 1]

print ('shape x1', np.shape(X1))
print ('shape x2', np.shape(X2))


# plot X1
ax.scatter(X1[0,:], X1[1,:],X1[2,:],  c='r', marker='o')
# plot X2
ax.scatter(X2[0,:], X2[1,:], X2[2,:],  c='b', marker='o')
ax.set_title(' 3D GMM')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.show()

# # ax.scatter(X1[0,:], X1[1,:], Gaussain_pdf(X1, var_x, var_y, corr, mean), c='r', marker='o')
# # ax.scatter(X2[0,:], X2[1,:], Gaussain_pdf(X2, var_x2, var_y2, corr2, mean2 ), c='b', marker='o')

# z = Gaussain_pdf(X1, var_x, var_y, corr, mean)
# # x = np.linspace(z.min(), z.np.max(), )
# # y = np.linspace(z.min(), z.max, 91)

# ax.plot_surface(x, y, z, rstride=4, cstride=4, linewidth=0)

# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('pdf')
# plt.show()




