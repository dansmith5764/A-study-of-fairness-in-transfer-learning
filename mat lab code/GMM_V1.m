clear 
close all 
clc


%create a probility distrobution for two classes 
P = [0.1 0.1 0.7 0.1];
%P = [0.4 0.6];
% assign mean to the each class
mu = {[0 1] [1 2]  [2 3] [4 5]};
%mu = {[0 0] [0 1]};
cov1 = chol([1 0; 0 1]);
cov2 = chol([1 0; 0 1]);
cov3 = chol([1 0; 0 1]);
cov4 = chol([1 0; 0 1]);

%assign a covari}nace matrix for each class
cov  = {cov1 cov2 cov3 cov4};
disp(cov{1})
%cov= {[1 0; 0 1]  [1 0; 0 1]};
%
Class = {0,1,2,3}; 
%Class = {0,1}; 

%%errror
colormap  = {'b o' '*g' 'k s' 'r +'};




N = 1000;
MC = 10;

for ii= 1:1:MC
    %create a matrix of bivarant gaussian variables 
    X= data_generation(N, mu, cov, P);
    idx_real = X(:,3);
    
    %% k-means magic
        [idx,C] = kmeans(X,length(Class),'Distance','cityblock');
    %%error
         [W,M,V,L] = mixGaussEm(X,length(Class),[],[],[],[]);

    for i = 1:length(Class)
        counts_real = length(X((idx_real==i),3));
        %counts = length(X((idx==i),3));
        counts = histcounts(idx_real((idx==i)),'BinMethod','integers','BinLimits',[1,length(Class)]);
        [maxCount, maxIdx] = max(counts);
        cluster(i) = i; 
        dominantGroup(i) = maxIdx;
        percentSame(i) = maxCount/sum(counts);
      
   
%          count_h_n(i)=sum(counts_h);
%         % count_n(i)=sum(counts);
%          count_real_n(i) =sum(counts_real); 
%          
    end
        
        totat_accuracy(ii)= sum(percentSame)/length(Class);
        
        
end


%%%%Methods of testing difference is distrobutions 
%%%%
accuracy  =  sum(totat_accuracy)/MC;
%KLD = sum(KLD)/MC;

% accuracy_KLD = KLDiv(y, y_k);


%plot sytic data 
figure; 
plot(X(idx_real==1,1), X(idx_real==1,2), 'b o');
hold on
plot(X(idx_real==2,1), X(idx_real==2,2), '*g');
hold on
plot(X(idx_real==3,1), X(idx_real==3,2), 'k s');
hold on
plot(X(idx_real==4,1), X(idx_real==4,2), 'r +');
legend('Cats','Dogs','Mice','Other',...
       'Location','NW')
n = num2str(N);
title('syntic data   N=' ,n);



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
 
% for i=1:1:MC
%     
%     
%         value  = mean(accuracy(2:i));
%         mean_ten(i) = value ;  
% end
% 
% figure;
% plot( 1:1:length(mean_ten),mean_ten, 'k.');
% xlabel('N');
% ylabel('mean of accuracy');
% title('Monty carlo simulation')


rng('default')  % For reproducibility

%%probs




[y, y_k] = PDF(Class, idx, idx_real, mu, cov,N,X);



%Plot the probability density values. ;
figure;
for i=1:1:length(Class)
    scatter3(X(idx==i,1),X(idx==i,2),y_k{i},colormap{i})
    hold on
end

xlabel('X1')
ylabel('X2')
zlabel('Probability Density ')
title('GMM k-means');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Location','NW')
hold off

%Plot the probability density values. ;
figure;
for i=1:1:length(Class)
    scatter3(X(idx_real==i,1),X(idx_real==i,2),y{i},colormap{i})
    hold on
end

xlabel('X1')
ylabel('X2')
zlabel('Probability Density ')
title('GMM ');
hold off


%%Kld
% y = [y{1} ; y{2}; y{3}; y{4}];
% y_k = [y_k{1} ; y_k{2}; y_k{3}; y_k{4}];
% KLD = crossentropy(y,y_k);




