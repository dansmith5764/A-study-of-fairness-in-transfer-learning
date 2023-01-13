clear 
close all 
clc


%% generate random data with fixed mean and covarinace 

%% other, cat , dog , mouse 
mu = [0 1 2 3];
mu_y= [0 1 2 3];
prob = [0.25 0.25 0.25 0.25]; %% probility distrution of C
if(sum(prob)~= 1)
    disp('error prob greater then 1')
    return
end 
sigma = [1 0 0 0 ;
         0 1 0 0 ;
         0 0 1 0;
         0 0 0 1];
Class = [0,1,2,3];
rows= 100;
collums= 4;
Num_points= rows*collums;
size = Num_points;

iterations  = 100;


%% matrix factorization 
%R  = chol(sigma);





%% loop this to test the classifier 

for ii=1:1:iterations
    %% create a matrix of randomly assigned classes with probilities
    Class_matrix = class_probility(size,prob);
    
    X= gaussian_distribution(mu,mu_y,sigma, Class_matrix);
    ind_mu = 0;
    
    
    % X= [cat ; dog; mouse; other];
    
    %%number of each class
    idx_c = find(X(:,3) == 1); %% cats are circles and 1's
    idx_d = find(X(:,3) == 2); %% dogs are starts and 2's
    idx_m = find(X(:,3) == 3); %% mice are squares and 3's
    idx_o = find(X(:,3) == 0); %% other are plus and 0's
    idx_real = [length(idx_c), length(idx_d), length(idx_m), length(idx_o)];
    
    
    
    
    
    %% k-means magic
    opts = statset('Display','final');
    [idx,C] = kmeans(X,length(Class),'Distance','cityblock',...
        'Replicates',5,'Options',opts);
    
T = array2table(zeros(length(Class),3),'VariableName',{'cluster','dominantGroup','percentSame'}); 

for i = 1:length(Class)
    counts = histcounts(X((idx==i),3),'BinMethod','integers','BinLimits',[1,length(Class)]);
    [maxCount, maxIdx] = max(counts);
    T.cluster(i) = i; 
    T.dominantGroup(i) = maxIdx;
    T.percentSame(i) = maxCount/sum(counts);
    
end
total_error(ii) = sum(T.percentSame(:), 'omitnan')/length(Class); 
disp(T)
    
    
    %%work out the error 
    
    %num_c_pre = find(idx == find(ind_mu==2)); %% difference(num_c vs num_c_pre ) is error 
    %error = abs(length(num_c_pre)-length(idx_c)); 
    
    %error_tot(ii) = error;
end

%%stats%%

% avg_error = 1- (sum(error_tot)/length(error_tot))/length(X);
avg_error = sum(total_error(:))/length(total_error);
disp('the average accurcy ');
disp(avg_error)


%plot sytic data 
figure; 
plot(X(idx_c,1), X(idx_c,2), 'b o');
hold on
plot(X(idx_d,1), X(idx_d,2), '*g');
hold on
plot(X(idx_m,1), X(idx_m,2), 'k s');
hold on
plot(X(idx_o,1), X(idx_o,2), 'r +');
legend('Cats','Dogs','Mice','Other',...
       'Location','NW')
title('syntic data');

%print the predication with K means 

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12) %other
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12) %cat
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12) %dog
hold on
plot(X(idx==4,1),X(idx==4,2),'k.','MarkerSize',12) %mouse
plot(C(:,1),C(:,2),'mx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off
