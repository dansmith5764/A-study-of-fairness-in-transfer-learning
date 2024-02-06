import numpy as np
import ot
from disparate import *
from utils import simulate_dataset, format_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from numpy.linalg import norm

def KLD(P, Q, epsilon): 
    # a small positive constant
    _P = (P + epsilon) / np.linalg.norm(P + epsilon, ord=1) 
    _Q = (Q + epsilon) / np.linalg.norm(Q + epsilon, ord=1) 
    return entropy(_P, _Q)


def modified_cholesky(A, alpha):
    """Computes the modified Cholesky decomposition of a matrix A.
    
    Parameters:
    A (numpy.ndarray): The matrix to decompose.
    alpha (float): A small positive constant to add to the diagonal to ensure positive definiteness.
    
    Returns:
    L (numpy.ndarray): The lower triangular matrix of the modified Cholesky decomposition.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        L[j,j] = np.sqrt(max(A[j,j] - np.sum(alpha*L[j,:]**2), 0))
        for i in range(j+1, n):
            L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j])) / (L[j,j] + alpha)
    return L

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


def merger_repair(X0,X1, alpha): 

    #find the mean and covariance of the two datasets
    mu = np.mean(X0, axis=0)
    cov = np.cov(X0.T)
    mu2 = np.mean(X1, axis=0)
    cov2 = np.cov(X1.T)

    #find the weights of the two datasets
    weights = np.array([X0.shape[0], X1.shape[0]])
    weights = weights/np.sum(weights)

    mu_m, cov_m, weights_m  = merge(mu, mu2, cov , cov2, weights)

    # A_m = np.linalg.cholesky(cov_m)
    A_m = modified_cholesky(cov_m, alpha=1e-6)
    d = np.shape(mu_m)[0]


    X0r = np.zeros((len(X0),d)) 
    for i in range (len(X0)):

            r = np.random.rand()
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
            


    X1r = np.zeros((len(X1),d))

    for i in range (len(X1)):
            
            r = np.random.rand()
            if r < alpha:
                    #step 1: zero the mean
                    x = X1[i,:] - mu2
                    #step 2: rotate the data
                    x = np.dot(A_m, x)
                    #step 3: scale the data
                    x = mu_m + x
                    X1r[i,:] = x
            else:
                    X1r[i,:] = X1[i,:]

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



def DI_list_merge_repair(X0, X1, beta0, beta1, iter):
    DIs = []
    accuracys = []
    F1s = []
    alphas = np.linspace(0,1,iter)
    for alpha in alphas:
        X0r, X1r = merger_repair(X0, X1, alpha)
        # log odds of positive outcome
        y0 =sigmoid(X0r.dot(beta0))
        # log odds of positive outcome
      
        y1 = sigmoid(X1r.dot(beta1))
        X,Y = format_dataset(X0r, X1r, y0, y1)

    
        #train logistic regression
        clf = LogisticRegression(random_state=69).fit(X[:, 1:],(Y>0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        accuracy = clf.score(X[:, 1:],(Y>0.5).astype(int))
        F1 = f1_score((Y>0.5).astype(int), Y_pred)


        DIs.append(disparate(X,Y_pred,0))
        accuracys.append(accuracy)
        F1s.append(F1)
    return DIs, accuracys, F1s


def DI_list_merge_repair_data(X0, X1, Y0, Y1,iter, model):
       
    lambdas = np.linspace(0,1,iter)
    DIs = []
    accuracys = []
    Classificatoin_error = []
    KLDs = []

    X0r_full, X1r_full = merger_repair(X0, X1, 1)
    X_full,Y_full = format_dataset(X0r_full, X1r_full, Y0, Y1)



    for lmbd in lambdas:
        X0r, X1r = merger_repair(X0, X1, lmbd)
       
  

    
        X,Y = format_dataset(X0r, X1r, Y0, Y1)


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=69)
        X_train = X_train[:, 1:]



        #train logistic regression model on train data
        # clf = LogisticRegression()
        model.fit(X_train, y_train)
        # predict on test data
        Y_pred = model.predict(X_test[:, 1:])
        Y_pred_full = model.predict(X_full[:, 1:])

        # #confusion matrix
        M = metrics.confusion_matrix(y_test, Y_pred)
        M1 = metrics.confusion_matrix(Y_full, Y_pred_full)

        # #KLD
        eps = 1e-15
        kld = KLD(M,M1, eps)


        DIs.append(disparate(X_test,Y_pred,0))
        accuracys.append(metrics.accuracy_score(y_test, Y_pred))
        Classificatoin_error.append(1 - metrics.accuracy_score(y_test, Y_pred))
        KLDs.append(kld)

        # accuracys.append(accuracy)
        # MSEs.append(MSE)
        # f1s.append(f1)

    return DIs, X0r, X1r, Y0, Y1, accuracys, Classificatoin_error, KLDs

from sklearn import metrics

def DI_list_geometric_repair_data(X0, X1, Y0, Y1,iter, model):
       
    lambdas = np.linspace(0,1,iter)
    DIs = []
    DIS_old = []
    accuracys = []
    Classificatoin_error = []
    KLDs = []

    X0r_full, X1r_full = geometric_repair(X0, X1, 1)
    X_full,Y_full = format_dataset(X0r_full, X1r_full, Y0, Y1)


    X_old, Y_old = format_dataset(X0, X1, Y0, Y1)
    X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, Y_old, test_size=0.4, random_state=69)

    #train logistic regression model on train data
    old_model = LogisticRegression()
    old_model.fit(X_train_old[:, 1:], y_train_old)



    for lmbd in lambdas:
        X0r, X1r = geometric_repair(X0, X1, lmbd)
       
  
        #train witht the repaired data
        X,Y = format_dataset(X0r, X1r, Y0, Y1)
        


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=69)
        
        X_train = X_train[:, 1:]



        #train logistic regression model on train data
        # clf = LogisticRegression()
        model.fit(X_train, y_train)


        # predict on with the non repaired data
        Y_pred = old_model.predict(X_test[:, 1:])
        Y_pred_full = old_model.predict(X_test_old[:, 1:])

        # # #confusion matrix
        # M = metrics.confusion_matrix(y_test, Y_pred)
        # print(M)
        # M1 = metrics.confusion_matrix(y_test_old, Y_pred_full)
        # print(M1)

        # # #KLD
        # eps = 1e-15
        # kld = KLD(M,M1, eps)

        # #acuracy
        # print(metrics.accuracy_score(y_test, Y_pred))

        # #classification error
        # print(1 - metrics.accuracy_score(y_test, Y_pred))

       
        


        DIs.append(disparate(X_test,Y_pred,0))
        DIS_old.append(disparate(X_test_old,Y_pred_full,0))
        accuracys.append(metrics.accuracy_score(y_test, Y_pred))
        Classificatoin_error.append(1 - metrics.accuracy_score(y_test, Y_pred))
        # KLDs.append(kld)

        # accuracys.append(accuracy)
        # MSEs.append(MSE)
        # f1s.append(f1)

    return DIs, X0r, X1r, Y0, Y1, accuracys, Classificatoin_error, KLDs, DIS_old


def DI_list_random_repair_data(X0, X1, Y0, Y1,iter, model):
       
    lambdas = np.linspace(0,1,iter)
    DIs = []
    accuracys = []
    Classificatoin_error = []



    for lmbd in lambdas:
        X0r, X1r = random_repair(X0, X1, lmbd)
       
  

    
        X,Y = format_dataset(X0r, X1r, Y0, Y1)


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=69)
        X_train = X_train[:, 1:]



        #train logistic regression model on train data
        # clf = LogisticRegression()
        model.fit(X_train, y_train)
        # predict on test data
        Y_pred = model.predict(X_test[:, 1:])

        # #confusion matrix
        # print(metrics.confusion_matrix(y_test, Y_pred))



        # #acuracy
        # print(metrics.accuracy_score(y_test, Y_pred))

        # #classification error
        # print(1 - metrics.accuracy_score(y_test, Y_pred))

       
        


        DIs.append(disparate(X_test,Y_pred,0))
        accuracys.append(metrics.accuracy_score(y_test, Y_pred))
        Classificatoin_error.append(1 - metrics.accuracy_score(y_test, Y_pred))

        # accuracys.append(accuracy)
        # MSEs.append(MSE)
        # f1s.append(f1)

    return DIs, X0r, X1r, Y0, Y1, accuracys, Classificatoin_error




def sigmoid(x):
    #limit the value of x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def DI_list_geometric_repair(X0, X1, beta0, beta1, iter):
    
    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)


    # geometric_repair(X0, X1, lmbd) - 
    # This function takes in two datasets X0 and X1 along with a scalar lmbd as input.
    #  It uses optimal transport to compute a linear mapping between the two datasets, 
    # computes the barycenter for each dataset, 
    # and then returns a repaired version of the datasets by combining the barycenter with the original datasets using lmbd as a weight.

    # Inputs:

    # X0: A numpy array representing the first dataset.
    # X1: A numpy array representing the second dataset.
    # lmbd: A scalar representing the weight to apply to the barycenter when combining with the original dataset.
    # Outputs:

    # X0_repaired: A numpy array representing the repaired version of X0.
    # X1_repaired: A numpy array representing the repaired version of X1.

    def geometric_repair(X0, X1, lmbd):
        w0,w1 = X0.shape[0], X1.shape[1]
        w0,w1 = w0/(w0+w1),w1/(w0+w1) # weights for barycenter, probility of each dataset

        barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0 # barycenter of X0
        barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

        X0_repaired = lmbd*(barycenter_0) + (1-lmbd)*X0 
        # repaired X0 = lmbd * barycenter of X0 + (1-lmbd) * X0
        X1_repaired = lmbd*(barycenter_1) + (1-lmbd)*X1
        # repaired X1 = lmbd * barycenter of X1 + (1-lmbd) * X1
        return  X0_repaired, X1_repaired


    lambdas = np.linspace(0,1,iter)
    DIs = []
    accuracys = []
    F1s = []


    # beta0 = (1, -1, -0.5, 1, -1)
    # beta1 = (1, -0.4, 1, -1, 1)

    for lmbd in lambdas:
        X0r, X1r = geometric_repair(X0, X1, lmbd)

        # log odds of positive outcome
        y0 =sigmoid(X0r.dot(beta0))
        # log odds of positive outcome
      
        y1 = sigmoid(X1r.dot(beta1))
        X,Y = format_dataset(X0r, X1r, y0, y1)

    
        #train logistic regression
        clf = LogisticRegression(random_state=69).fit(X[:, 1:],(Y>0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        accuracy = clf.score(X[:, 1:],(Y>0.5).astype(int))
        F1 = f1_score((Y>0.5).astype(int), Y_pred)
        

       #forst 10 lines of X
        print(X[0,:])
        print(Y[0])
        DIs.append(disparate(X,Y_pred,0))
        accuracys.append(accuracy)
        F1s.append(F1)
    return DIs, X0r, X1r, y0, y1, accuracys, F1s


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

def DI_list_random_repair(X0, X1, beta0, beta1):
    
    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    def random_repair(X0, X1, lmbd):
        w0,w1 = X0.shape[0], X1.shape[1]
        w0,w1 = w0/(w0+w1),w1/(w0+w1)

        barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
        barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

        ber0, ber1 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1)), np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

        X0_repaired = ber0*(barycenter_0) + (1-ber0)*X0
        X1_repaired = ber1*(barycenter_1) + (1-ber1)*X1
        return  X0_repaired, X1_repaired

    lambdas = np.linspace(0,1,1000)
    DIs = []

    # beta0 = (1, -1, -0.5, 1, -1)
    # beta1 = (1, -0.4, 1, -1, 1)

    for lmbd in lambdas:
        X0r, X1r = random_repair(X0, X1, lmbd)
        # print(X0r.sum())

        y0 = np.exp(X0r.dot(beta0)) / (1 + np.exp(X0r.dot(beta0)))
        y1 = np.exp(X1r.dot(beta1)) / (1 + np.exp(X1r.dot(beta1)))
        X,Y = format_dataset(X0r, X1r, y0, y1)

        # X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

        # print((Y>0.5).astype(int))

        clf = LogisticRegression(random_state=69).fit(X[:, 1:],(Y>0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        DIs.append(disparate(X,Y_pred,0)[1])
    return DIs