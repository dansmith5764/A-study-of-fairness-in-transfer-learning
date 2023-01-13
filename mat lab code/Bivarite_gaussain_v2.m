clear 
close all 
clc


%create a probility distrobution for two classes 
P = [0.25 0.25 0.25 0.25];
% assign mean to the each class
mu = {[0 1] [1 2]  [2 3] [3 4]};
%assign a covarinace matrix for each class
cov= {[1 0; 0 1]  [1 0; 0 1] [1 0; 0 1]  [1 0; 0 1]};
%
Class = {0,1,2,3}; 
%%errror


N = 500;
MC = 1;

for ii= 1:1:MC
    %create a matrix of bivarant gaussian variables 
    X= data_generation(N, mu, cov, P);
    idx_real = X(:,3);
    
    %% k-means magic
%         opts = statset('Display','final');
%         [idx,C] = kmeans(X,length(Class),'Distance','cityblock',...
%             'Replicates',5,'Options',opts);
%opts = statset('Display','final');
        [idx,C] = kmeans(X,length(Class),'Distance','cityblock');


    
    %%error
    for i = 1:length(Class)
        counts = histcounts(X((idx==i),3),'BinMethod','integers','BinLimits',[1,length(Class)]);
        [maxCount, maxIdx] = max(counts);
         
         count_n(i)=sum(counts);
         sam(i) = maxCount;
         if(sum(counts) == 0)
             percentSame(i) = 0;
         else
            percentSame(i) = maxCount/sum(counts);
         end
       
        
    end
    if(length(Class) == 0)
        disp('huh');
    else
        all_percentage_same(ii) = sum(percentSame)/length(Class);
        if(isnan(all_percentage_same(ii)))
            disp('double huh')
        end 
        same(ii)= sum(sam)/length(Class);
        totat_error(ii) = sum(percentSame)/length(Class);
    end 
     
end

%r = sum((idx_real-idx)^2)/MC;
bias  = sum(same)/MC;
accuracy  = sum(totat_error)/MC;


%plot sytic data 
figure; 
plot(X(idx_real==0,1), X(idx_real==0,2), 'b o');
hold on
plot(X(idx_real==1,1), X(idx_real==1,2), '*g');
hold on
plot(X(idx_real==2,1), X(idx_real==2,2), 'k s');
hold on
plot(X(idx_real==3,1), X(idx_real==3,2), 'r +');
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
 
for i=1:1:MC
    
    
        value  = mean(all_percentage_same(2:i));
        mean_ten(i) = value ;  
end

figure;
plot( 1:1:length(mean_ten),mean_ten, 'k.');
xlabel('N');
ylabel('mean of accuracy');
title('Monty carlo simulation')
