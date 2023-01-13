clear 
close all 
clc


mu = [1 -1];
Sigma = [0.9 0.4; 0.4 0.3];
%Randomly sample from the distribution 100 times. Specify X as the matrix of sampled points.

rng('default')  % For reproducibility
X = mvnrnd(mu,Sigma,100);
%Evaluate the pdf of the distribution at the points in X.

y = mvnpdf(X,mu,Sigma);
%Plot the probability density values.
%X1 -2 5
x1 = -2 : (2+5)/100:5 -(2+5)/100
%x2 -1.5 -2.4
x2 = -1.5: -(2.5-1.5)/100: -2.5 +(2.5-1.5)/100
% 0 t0 0.5
x3 = 0: 0.5/100:0.5-0.5/100

Z = meshgrid(1:3,x1);
M = meshgrid(x2,1:3);
m = [X(:,1) X(:,2) y ];
%scatter3(X(:,1),X(:,2),y)
surf(Z,M,m)
xlabel('X1')
ylabel('X2')
zlabel('Probability Density')