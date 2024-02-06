import numpy as np
import matplotlib.pylab as plt
import ot

def simulate_dataset(n0, n1, mu0, mu1, sigma, beta0, beta1):
  # Sample examples from the two multivariate normal distributions
  X0 = np.random.multivariate_normal(mu0, sigma, size=n0)
  X1 = np.random.multivariate_normal(mu1, sigma, size=n1)
  
  # Compute the logit model for each group
  logit0 = np.exp(X0.dot(beta0)) / (1 + np.exp(X0.dot(beta0)))
  logit1 = np.exp(X1.dot(beta1)) / (1 + np.exp(X1.dot(beta1)))
  
  # Compute the classification labels
  Y0 = (logit0 > 0.5).astype(int)
  Y1 = (logit1 > 0.5).astype(int)
  
  return X0, X1, Y0, Y1

def format_dataset(X0, X1, Y0, Y1):

    
    X= np.concatenate([X0,X1]) # Concatenate the two groups
    Y= np.concatenate([Y0, Y1])
    S= np.concatenate([np.repeat(0, X0.shape[0]),np.repeat(1, X1.shape[0])]) # Create the sensitive attribute
    S= np.expand_dims(S, 1) # Add a dimension to S
    X= np.concatenate([S, X], axis=1) # Concatenate S and X
    return X, Y

def simulate_dataset_UF(n0, n1, mu0, mu1, mu2, mu3, sigma, beta0, beta1,  prob,prob1, d):
    
    X0	= np.zeros((n0,d))
    X1	= np.zeros((n1,d))
    Y0	= np.zeros((n0,1))
    Y1	= np.zeros((n1,1))
    for i in range(n0):
        r = np.random.rand() 
        if r <= prob:
            x0 = np.random.multivariate_normal(mu0, sigma, size=1)
            X0[i,:] = x0
       
        else:
            x0 = np.random.multivariate_normal(mu1, sigma, size=1)
            X0[i,:] = x0	
    
    for i in range(n1): 
        r = np.random.rand() 
        if r <= prob1:
            x1 = np.random.multivariate_normal(mu2, sigma, size=1)
            X1[i,:] = x1
      
        else:
            x1 = np.random.multivariate_normal(mu3, sigma, size=1)
            X1[i,:] = x1
     
    # Compute the logit model for each group
    logit0 = np.exp(X0.dot(beta0)) / (1 + np.exp(X0.dot(beta0)))
    logit1 = np.exp(X1.dot(beta1)) / (1 + np.exp(X1.dot(beta1)))

 
    # Compute the classification labels
    Y0 = (logit0 > 0.5).astype(int)
    Y1 = (logit1 > 0.5).astype(int)

  
    return X0, X1, Y0, Y1
