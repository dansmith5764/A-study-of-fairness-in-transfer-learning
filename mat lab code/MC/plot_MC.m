clear 
close all 
clc
addpath('..');

Class = {0,1}; 
mu = {[0 0] [1 1]};
P = [0.6 0.4];
iterations = 50;
size_interation=0.02;
MC = 1000;
N = 1000;
cov= {[1 0; 0 1]  [1 0; 0 1] };


%[accuracy_var, std_var] = BGM_MC_var_xy(P,MC, N, iterations,size_interation, Class,  mu);
[accuracy_means, std_mean,total_cross_loss] = BVG_MC_dist_means(P,MC, N, iterations,size_interation, Class,  cov);
%[accuracy_cov, std_cov] = BGM_MC_cov_xy(P,MC, N, iterations, Class,  size_interation,mu) ;
%l = length(accuracy_var);
% 
d = 0.02:0.02:1;
figure; 
hold on 
title('MC simulation')



plot(flip(d),accuracy_means, 'k')
plot(flip(d),accuracy_means +std_mean, 'r')
plot(flip(d),accuracy_means - std_mean, 'r')
xlabel('Distance between means')
ylabel('percentage misclassification(%)')
legend('Mean ', 'standard deviation')

hold off 

% figure; 
% hold on 


% title('MC simulation')
% plot(1:l,accuracy_cov, 'b')
% plot(1:l,accuracy_var, 'k')
% plot(1:l,accuracy_means, 'r')
% xlabel('n')
% ylabel('Accuracy')
% legend('Decreasing distance between means', 'increasing variance', 'increasing covariance')
% 
% hold off 

