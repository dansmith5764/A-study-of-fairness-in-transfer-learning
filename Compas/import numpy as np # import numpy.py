import numpy as np                              # import numpy
from numpy.linalg import inv                    # for matrix inverse
import matplotlib.pyplot as plt                 # import matplotlib.pyplot for plotting framework
from scipy.stats import multivariate_normal     # for generating pdf
import random

## data generation
m1 = [1,1]      # consider a random mean and covariance value
m2 = [7,7]                                              
cov1 = [[3, 2], [2, 3]]                                      
cov2 = [[2, -1], [-1, 2]]
x = np.random.multivariate_normal(m1, cov1, size=(200,))  # Generating 200 samples for each mean and covariance
y = np.random.multivariate_normal(m2, cov2, size=(200,))
d = np.concatenate((x, y), axis=0)


##Plotting the ground truth

# plt.figure(figsize=(10,10))                                 
# plt.scatter(d[:,0], d[:,1], marker='o')     
# plt.axis('equal')                                  
# plt.xlabel('X-Axis', fontsize=16)              
# plt.ylabel('Y-Axis', fontsize=16)                     
# plt.title('Ground Truth', fontsize=22)    
# plt.grid()            
# plt.show()

#Taking initial guesses for the parameters

m1 = random.choice(d)
m2 = random.choice(d)
cov1 = np.cov(np.transpose(d))
cov2 = np.cov(np.transpose(d))
pi = 0.5

#Plotting Initial State

x1 = np.linspace(-4,11,200)  
x2 = np.linspace(-4,11,200)
X, Y = np.meshgrid(x1,x2) 

Z1 = multivariate_normal(m1, cov1)  
Z2 = multivariate_normal(m2, cov2)


pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
pos[:, :, 0] = X; pos[:, :, 1] = Y   
print('shape of Z1', np.shape(Z1.pdf(pos)))


# plt.figure(figsize=(10,10))                                                          # creating the figure and assigning the size
# plt.scatter(d[:,0], d[:,1], marker='o')     
# plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
# plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
# plt.axis('equal')                                                                  # making both the axis equal
# plt.xlabel('X-Axis', fontsize=16)                                                  # X-Axis
# plt.ylabel('Y-Axis', fontsize=16)                                                  # Y-Axis
# plt.title('Initial State', fontsize=22)                                            # Title of the plot
# plt.grid()                                                                         # displaying gridlines
# plt.show()


#Expectation Step

##Expectation step
def Estep(lis1):
    m1=lis1[0]
    m2=lis1[1]
    cov1=lis1[2]
    cov2=lis1[3]
    pi=lis1[4]
    
    pt2 = multivariate_normal.pdf(d, mean=m2, cov=cov2)
    pt1 = multivariate_normal.pdf(d, mean=m1, cov=cov1)
    w1 = pi * pt2
    w2 = (1-pi) * pt1
    eval1 = w1/(w1+w2)

    return(eval1)


#Maximization Step

## Maximization step
def Mstep(eval1):
    num_mu1,din_mu1,num_mu2,din_mu2=0,0,0,0

    for i in range(0,len(d)):
        num_mu1 += (1-eval1[i]) * d[i]
        din_mu1 += (1-eval1[i])

        num_mu2 += eval1[i] * d[i]
        din_mu2 += eval1[i]

    mu1 = num_mu1/din_mu1
    mu2 = num_mu2/din_mu2

    num_s1,din_s1,num_s2,din_s2=0,0,0,0
    for i in range(0,len(d)):

        q1 = np.matrix(d[i]-mu1)
        num_s1 += (1-eval1[i]) * np.dot(q1.T, q1)
        din_s1 += (1-eval1[i])

        q2 = np.matrix(d[i]-mu2)
        num_s2 += eval1[i] * np.dot(q2.T, q2)
        din_s2 += eval1[i]

    s1 = num_s1/din_s1
    s2 = num_s2/din_s2

    pi = sum(eval1)/len(d)
    
    lis2=[mu1,mu2,s1,s2,pi]
    return(lis2)


#Function to plot the EM algorithm

def plot(lis1):
    mu1=lis1[0]
    mu2=lis1[1]
    s1=lis1[2]
    s2=lis1[3]
    Z1 = multivariate_normal(mu1, s1)  
    Z2 = multivariate_normal(mu2, s2)

    pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
    pos[:, :, 0] = X; pos[:, :, 1] = Y   

    plt.figure(figsize=(10,10))                                                          # creating the figure and assigning the size
    plt.scatter(d[:,0], d[:,1], marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
    plt.axis('equal')                                                                  # making both the axis equal
    plt.xlabel('X-Axis', fontsize=16)                                                  # X-Axis
    plt.ylabel('Y-Axis', fontsize=16)                                                  # Y-Axis
    plt.grid()                                                                         # displaying gridlines
    plt.show()

#Calling the functions and repeating until it converges

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


iterations = 20
lis1=[m1,m2,cov1,cov2,pi]
for i in range(0,iterations):
    lis2 = Mstep(Estep(lis1))
    lis1=lis2
    if(i==0 or i == 4 or i == 9 or i == 14 or i == 19):
        #plot(lis1)
        m1=lis1[0]
        m2=lis1[1]
        cov1=lis1[2]
        cov2=lis1[3]
        pi=lis1[4]

       
       
        var_x1 = cov1[0,0]
        var_y1 = cov1[1,1]
        corr1 = cov1[0,0]/np.sqrt(var_x1*var_y1)
        var_x2 = cov2[0,0]
        var_y2 = cov2[1,1]
        corr2 = cov2[0,1]/np.sqrt(var_x2*var_y2)


        print('shape of X', np.shape(X))
        print('shape of Y', np.shape(Y))

        print('X', X[0,:])

        Z1 = multivariate_normal(m1, cov1)
        Z2 = multivariate_normal(m2, cov2)

        pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
        pos[:, :, 0] = X; pos[:, :, 1] = Y   

        #print first 10 values of Z1.pdf(pos)
        print('Z1.pdf(pos)', Z1.pdf(pos)[0:10,0:10])
        print('Z2.pdf(pos)', Z2.pdf(pos)[0:10,0:10])



       
    

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1, projection='3d')

        

        # #plot the original data
        # ax.scatter(X[0,:], X[1,:], Gaussain_pdf(X[0,:], X[1,:], var_x1, var_y1, corr1, m1 ), c='r', marker='o')
        # ax.scatter(Y[0,:], Y[1,:], Gaussain_pdf(Y[0,:], Y[1,:], var_x2, var_y2, corr2, m1 ), c='r', marker='o')
        # ax.set_xlabel('x1')
        # ax.set_ylabel('x2')
        # ax.set_zlabel('pdf')
        # ax.set_title('Original Data')
        # plt.show()