clear 
close all 
clc


%create a probility distrobution for two classes 
P = [0.6 0.4];

%assign a covari}nace matrix for each class
cov= {[1 0; 0 1]  [1 0; 0 1] };

% assign mean to the each class
    c1 = 0;
    c2 = 0;

for j=1:1:50
  
    cov_tot{j} = {[1 0; 0 1]  [1 c1; c2 1]};
    c1 = c1+.02;
    c2 = c2+.02;
end 


for i=1:1:length(cov_tot)

    disp(i);
    disp(cov_tot{i}{2})
end 

%mu_tot = {mu1 mu2 mu3 mu4 mu5mu1 mu2 mu3 mu4 mu5 mu6 mu7 mu8 mu9 mu10 };



%
Class = {0,1}; 
mu = {[0 1] [1 2]};
%%errror
colormap  = {'b o' '*g' 'k s' 'r +'};



N = 1000;
MC = 10;

%% loop for differnt opperating contionous 
for iii =1:1:length(cov_tot)

    cov = cov_tot{iii};
    for ii= 1:1:MC
        %create a matrix of bivarant gaussian variables 
        X= data_generation(N, mu, cov, P);
        idx_real = X(:,3);
        
        %% k-means magic
            [idx,C] = kmeans(X,length(Class),'Distance','cityblock');
        %%error
        for i = 1:length(Class)
        counts = histcounts(idx_real((idx==i)),'BinMethod','integers','BinLimits',[1,length(Class)]);
        [maxCount, maxIdx] = max(counts);
        percentSame(i) = maxCount/sum(counts);
        end
            totat_accuracy(ii)= sum(percentSame)/length(Class);

        %% EM magic
        


        %%get the Cross entropy loss 
        [y, y_k] = PDF(Class, idx, idx_real, mu, cov,N,X);
        y = [y{1} ; y{2}];
        y_k = [y_k{1} ; y_k{2}];
        KLD(ii) = crossentropy(y,y_k);
    end

    %%histcounts loss
    accuracy_iterations(iii)  =  sum(totat_accuracy)/MC;
    %%% KDL loss 
    KDL_total(iii)= sum(KLD)/MC;
end
accuracy  =  sum(totat_accuracy)/MC;


% plot the average value from all the simulations 
    figure;
    hold on
    title('monty carlo simulation');
    plot( 1:length(cov_tot),accuracy_iterations);
    %plot(1:length(mu_tot), KDL_total, 'r');
    hold off

% plot sytic data 
% figure; 
% plot(X(idx_real==1,1), X(idx_real==1,2), 'b o');
% hold on
% plot(X(idx_real==2,1), X(idx_real==2,2), '*g');
% hold on
% plot(X(idx_real==3,1), X(idx_real==3,2), 'k s');
% hold on
% plot(X(idx_real==4,1), X(idx_real==4,2), 'r +');
% legend('Cats','Dogs','Mice','Other',...
%        'Location','NW')
% n = num2str(N);
% title('syntic data   N=' ,n);
% 
% 
% 
% print the predication with K means 
% 
% figure;
% plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12) %other
% hold on
% plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12) %cat
% hold on
% plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12) %dog
% hold on
% plot(X(idx==4,1),X(idx==4,2),'k.','MarkerSize',12) %mouse
% plot(C(:,1),C(:,2),'mx',...
%      'MarkerSize',15,'LineWidth',3) 
% legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
%        'Location','NW')
% title 'Cluster Assignments and Centroids'
% hold off



