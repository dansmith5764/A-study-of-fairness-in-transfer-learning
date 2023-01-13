import numpy as np 

def KLD_Gaussians(mu_1, sigma_1, mu_2, sigma_2, d):
    #1/2*[log|Σ2|/|Σ1|−d+tr{Σ2^-1*Σ1}+(μ2−μ1)^T*Σ2^−2(μ2−μ1)].

    #KL(P||Q) = -1/2 * [log(|2*pi*sigma_p|) + log(|sigma_q|) + (mu_p – mu_q)’*sigma_q^-1*(mu_p – mu_q) + trace(sigma_p^-1*sigma_q)]


    det_1 = np.linalg.det(sigma_1)
    det_2 = np.linalg.det(sigma_2)
    print(np.shape(sigma_1))

    b = np.trace(np.multiply(np.linalg.inv(sigma_2), sigma_1))
    a = np.log(det_2) - np.log(det_1)
    c = np.transpose(mu_2 - mu_1)*np.dot(np.linalg.inv(sigma_2),(mu_2 - mu_1))
 

    return  np.multiply(0.5,(a + b + c - d))


   
mu_1 = np.array([0, 00, 0])
sigma_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
mu_2 = np.array([0, 0, 0])
sigma_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
d = 3
print(KLD_Gaussians(mu_1, sigma_1, mu_2, sigma_2,  d))
