clear 
close all 
clc


mu = [0 0];
Sigma = [0.25 0.3; 0.3 1];


%%% Compute the probability over the unit square.
p = mvncdf([0 0],[1 1],mu,Sigma);

%% create a grid of evenly spaced points in two-dimensional space.
x1 = -3:.2:3;
x2 = -3:.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

%%evulate pdf of the grid
y = mvnpdf(X,mu,Sigma);
y = reshape(y,length(x2),length(x1));


contour(x1,x2,y,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35])
xlabel('x')
ylabel('y')
line([0 0 1 1 0],[1 0 0 1 1],'Linestyle','--','Color','k')