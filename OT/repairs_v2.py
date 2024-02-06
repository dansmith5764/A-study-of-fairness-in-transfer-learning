import numpy as np
import ot
from disparate import *
from utils import simulate_dataset, format_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt


blue1 = '#1f77b4' # darker blue
blue2 = '#aec7e8' # lighter blue

# Two shades of red
red1 = '#d62728' # darker red
red2 = '#ff9896' # lighter red

light_green = '#90EE90'
dark_green = '#006400'

import itertools

from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.stats import entropy
from numpy.linalg import norm

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X= gmm_p.score_samples(X)
    log_q_X= gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def KLD(P, Q, epsilon): 
    # a small positive constant
    _P = (P + epsilon) / np.linalg.norm(P + epsilon, ord=1) 
    _Q = (Q + epsilon) / np.linalg.norm(Q + epsilon, ord=1) 
    return entropy(_P, _Q)

def fair_clustering(X_F,y_A ):
    
    X_F  = X_F[:,0:2]
    X_train, X_test, y_train, y_test = train_test_split(X_F, y_A, test_size=0.3, random_state=42)


    n_com = np.unique(y_A).shape[0]
    
    gmm = GaussianMixture(n_components=n_com, covariance_type='full', random_state=0).fit(X_train)
    labels_gmm = gmm.predict(X_test)


    M = np.zeros((n_com,n_com))
    for j in range(n_com):
        for i in range(n_com):
            M[i,j] = X_test[(y_test == i) & (labels_gmm == j), 1].shape[0]

    N = np.repeat(M, 2, axis=1)
    # print (N)
    N = N/N.sum(axis=0)
    # print (N)

    # #plot X_test and labels_gmm
    # fig = plt.figure(figsize=(8, 8))

    # colors = [blue1, blue2]
    
    # for i in range(2):
    #     plt.scatter(X_test[labels_gmm==i, 0], X_test[labels_gmm==i, 1], label=r'$\hat{Y}$ ='+str(i), color=colors[i], alpha=0.5)
    # plt.legend()
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.show()

    return N, labels_gmm, y_test, gmm

def merge(mu, mu2, cov , cov2, weights):
    # merge two Gaussian
    # Input:
    # mu, mu2: means of the two Gaussians
    # cov, cov2: covariance matrices of the two Gaussians
    # weights: weights of the two Gaussians
    # Output:
    # mu_m: mean of the merged Gaussian
    # cov_m: covariance matrix of the merged Gaussian
    # weight_m: weight of the merged Gaussian
    # Reference: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Merging_of_normal_distributions
    # Written by Mo Chen (

    alpha = (weights[0])**2/(weights[0]+weights[1])**2
    #alpha = (weights[0])/(weights[0]+weights[1])

    mu_m = (weights[0]*mu + weights[1]*mu2)/(weights[0]+weights[1])
    #cov_m = (weights[0]*cov + weights[1]*cov2 + weights[0]*weights[1]/(weights[0]+weights[1])*(mu-mu2).dot((mu-mu2).T))/(weights[0]+weights[1])
    cov_m  = alpha*cov + (1-alpha)*cov2
    weight_m = weights[0]+weights[1]

    return mu_m, cov_m, weight_m


def m_repair(X0, mu, mu_m, A_m, d, alpha) : 
    X0r = np.zeros((len(X0),d)) 
    for i in range (len(X0)):

            r = np.random.rand()
            # r = np.random.binomial
            if r < alpha:
                #step 1: zero the mean
                x = X0[i,:] - mu
                #step 2: rotate the data
                x = np.dot(A_m, x)
                #step 3: scale the data
                x = mu_m + x
                X0r[i,:] = x
            else:
                X0r[i,:] = X0[i,:]

    return X0r
     

def merger_repair(X0,X1, alpha):

    #find the mean and covariance of the two datasets
    mu = np.mean(X0, axis=0)
    cov = np.cov(X0.T)
    mu2 = np.mean(X1, axis=0)
    cov2 = np.cov(X1.T)

    #find the weights of the two datasets
    weights = np.array([X0.shape[0], X1.shape[0]])
    weights = weights/np.sum(weights)

    mu_m, cov_m, weights_m  = merge(mu, mu2, cov, cov2, weights)

    # A_m = np.linalg.cholesky(cov_m)
    A_m = np.linalg.cholesky(cov_m)
    d = np.shape(mu_m)[0]

    X0r = m_repair(X0, mu, mu_m, A_m, d, alpha)
    X1r = m_repair(X1, mu2, mu_m, A_m, d, alpha)

    return X0r, X1r

def geometric_repair(X0, X1,  lmbd):

    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    w0,w1 = X0.shape[0], X1.shape[1]
    w0,w1 = w0/(w0+w1),w1/(w0+w1)

    barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
    barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

    X0_repaired = lmbd*(barycenter_0) + (1-lmbd)*X0
    X1_repaired = lmbd*(barycenter_1) + (1-lmbd)*X1
    return  X0_repaired, X1_repaired


def DI_list_geometric_repair(X0, X1,iter, Y0, Y1, gmm_unfair):
   
    lambdas = np.linspace(0,1,iter)
    DIs = []
    klds = []
    klds_gmm = []
  

    count = 0

    for lmbd in lambdas:
        X0r, X1r = geometric_repair(X0, X1, lmbd)

     
        X = np.concatenate((X0r, X1r), axis=0)
        Y = np.concatenate((Y0, Y1), axis=0)
           

        y_Au = Y[:, 0]
        y_Ap = Y[:, 1] 
        N, labels_gmm_F, y_test, gmm = fair_clustering(X, y_Au)
    


        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        y_test_1 = y_test[:,0] #unprotected
        y_test_2 = y_test[:,1] #protected

        #predict the labels
        labels_gmm = gmm.predict(X_test)


        # print('ARI', adjusted_rand_score(labels_gmm, labels_gmm_F))
      
       
       
    
        #we need to see the result of the cluster and the protected attribute
        XY = np.concatenate((X_test[:,0].reshape(-1,1), X_test[:,1].reshape(-1,1), y_test_1.reshape(-1,1), y_test_2.reshape(-1,1), labels_gmm.reshape(-1,1)), axis=1)
     
        dis = DI(XY)


        print(np.unique(labels_gmm))
        #stack labels_gmm with X_test
        X_test = np.concatenate(( labels_gmm.reshape(-1,1), X_test), axis=1)
        #print first 10 rows
        # print(X_test[:10,:])

        # print(X_test.shape)
        # print(labels_gmm.shape)
    

        y_Au = XY[:, 2] 
        y_Ap = XY[:, 3]
        y_pred = XY[:, 4]

        
        M = np.zeros((2,4))
        count = 0
        for i in range(2):
            for j in range(2):
                M[0,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 1), 4].shape[0]
                M[1,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 0), 4].shape[0]
                count +=1
        
        
        #normalize the matrix
        M = M/M.sum(axis=0)


        N_flip = np.flipud(N) 
        kld = KLD(M, N_flip, epsilon=1e-10)
        

        #kld between the gmm and the original
        kld_gmm = gmm_kl(gmm, gmm_unfair)
        klds_gmm.append(kld_gmm)


       
        DIs.append(dis)
        klds.append(kld)
       
    return DIs, X0r, X1r, klds, klds_gmm

def DI(XY):
    count_4_5 = np.sum((XY[:, 3] == 1) & (XY[:, 4] == 1))
    count_4_6 = np.sum((XY[:, 3] == 0) & (XY[:, 4] == 1))

    eps = 1e-10
    a = count_4_5/(count_4_6 + eps)
    b = count_4_6/(count_4_5 + eps)

    if a > b:
        di = b
    else:
        di = a

    return di
    
def random_repair(X0, X1, lmbd):

    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    w0,w1 = X0.shape[0], X1.shape[1]
    w0,w1 = w0/(w0+w1),w1/(w0+w1)

    barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
    barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

    ber0, ber1 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1)), np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

    X0_repaired = ber0*(barycenter_0) + (1-ber0)*X0
    X1_repaired = ber1*(barycenter_1) + (1-ber1)*X1
    return  X0_repaired, X1_repaired

def DI_list_random_repair(X0, X1,iter, Y0, Y1, gmm_unfair):

    lambdas = np.linspace(0,1,iter)
    DIs = []
    accuracys = []

    klds = []
    klds_gmm = []

    # beta0 = (1, -1, -0.5, 1, -1)
    # beta1 = (1, -0.4, 1, -1, 1)

    for lmbd in lambdas:
        X0r, X1r = random_repair(X0, X1, lmbd)
        # print(X0r.sum())

          
        X = np.concatenate((X0r, X1r), axis=0)
        Y = np.concatenate((Y0, Y1), axis=0)
           

        y_Au = Y[:, 0]
        y_Ap = Y[:, 1] 
        N, labels_gmm_F, y_test, gmm = fair_clustering(X, y_Au)
    


        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        y_test_1 = y_test[:,0] #unprotected
        y_test_2 = y_test[:,1] #protected

        #predict the labels
        labels_gmm = gmm.predict(X_test)

    
        #we need to see the result of the cluster and the protected attribute
        XY = np.concatenate((X_test[:,0].reshape(-1,1), X_test[:,1].reshape(-1,1), y_test_1.reshape(-1,1), y_test_2.reshape(-1,1), labels_gmm.reshape(-1,1)), axis=1)
     
        dis = DI(XY)


        print(np.unique(labels_gmm))
        #stack labels_gmm with X_test
        X_test = np.concatenate(( labels_gmm.reshape(-1,1), X_test), axis=1)
    
        y_Au = XY[:, 2] 
        y_Ap = XY[:, 3]
        y_pred = XY[:, 4]

        
        M = np.zeros((2,4))
        count = 0
        for i in range(2):
            for j in range(2):
                M[0,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 1), 4].shape[0]
                M[1,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 0), 4].shape[0]
                count +=1
        
        
        #normalize the matrix
        M = M/M.sum(axis=0)

        N_flip = np.flipud(N) 
        kld = KLD(M, N_flip, epsilon=1e-10)

        #kld between the gmm and the original
        kld_gmm = gmm_kl(gmm, gmm_unfair)
        klds_gmm.append(kld_gmm)
        DIs.append(dis)
        klds.append(kld)
       
    return DIs, X0r, X1r, klds, klds_gmm


def DI_list_merge_repair(X0, X1,iter, Y0, Y1, gmm_unfair):
    DIs = []
    accuracys = []
    alphas = np.linspace(0,1,iter)
    klds = []
    klds_gmm = []
    for alpha in alphas:
        X0r, X1r = merger_repair(X0, X1, alpha)

        X = np.concatenate((X0r, X1r), axis=0)
        Y = np.concatenate((Y0, Y1), axis=0)
           
        y_Au = Y[:, 0]
        y_Ap = Y[:, 1] 
        N, labels_gmm_F, y_test, gmm = fair_clustering(X, y_Au)
    


        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        y_test_1 = y_test[:,0]
        y_test_2 = y_test[:,1]

        #predict the labels
        labels_gmm = gmm.predict(X_test)


        # print('ARI', adjusted_rand_score(labels_gmm, labels_gmm_F))
      
       
       
    
        #we need to see the result of the cluster and the protected attribute
        XY = np.concatenate((X_test[:,0].reshape(-1,1), X_test[:,1].reshape(-1,1), y_test_1.reshape(-1,1), y_test_2.reshape(-1,1), labels_gmm.reshape(-1,1)), axis=1)
     
        dis = DI(XY)

    

        y_Au = XY[:, 2] 
        y_Ap = XY[:, 3]
        y_pred = XY[:, 4]

        
        M = np.zeros((2,4))
        count = 0
        for i in range(2):
            for j in range(2):
                M[0,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 1), 4].shape[0]
                M[1,count] = XY[(y_Au == i) & (y_Ap == j) & (y_pred == 0), 4].shape[0]
                count +=1
        
        
        #normalize the matrix
        M = M/M.sum(axis=0)


        N_flip = np.flipud(N) 
        kld = KLD(M, N_flip, epsilon=1e-10)

     
        #kld between the gmm and the original
        kld_gmm = gmm_kl(gmm, gmm_unfair)


        klds_gmm.append(kld_gmm)
        DIs.append(dis)
        klds.append(kld)
        

        



    return DIs, X0r, X1r, klds, klds_gmm