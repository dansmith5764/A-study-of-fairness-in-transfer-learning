clear 
close all 
clc

mu = [1 -1];
Sigma = [0.9 0.4; 0.4 0.3];

rng('default')  % For reproducibility
X = mvnrnd(mu,Sigma,100);
%Evaluate the pdf of the distribution at the points in X.

y = mvnpdf(X,mu,Sigma);
y = [X(:,1),X(:,2),y];
m1 = min(X(:,1));
m2 = max(X(:,1));
m3 = min(X(:,2));
m4 = max(X(:,2));

Y = meshgrid(-2:4/size(y,1):2 -(4/size(y,1)),1:size(y,2));
Y  = reshape(Y, 100,3);
%Plot the probability density values.
surf(Y,Y,y)
xlabel('X1')
ylabel('X2')
zlabel('Probability Density')
