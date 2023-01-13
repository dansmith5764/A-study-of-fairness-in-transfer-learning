clear 
close all 
clc


%% generate random data with fixed mean and covarinace 

%% other, cat , dog , mouse 
mu = [0 1.5 3 4.5 ];
mu_y= [1 2 3 4];
prob = [0.1 0.2 0.3 0.4]; %% probility distrution of C
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

iterations  = 10;


%% matrix factorization 
%R  = chol(sigma);





%% loop this to test the classifier 

for ii=1:1:iterations
    %% create a matrix of randomly assigned classes with probilities
    Class_matrix = class_probility(size,prob);
    
    X= gaussian_distribution(mu, mu_y,sigma, Class_matrix);
    
    
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
    
    %find the mean of each created cluster 
    ind_mu=5;
    for i=1:1:length(C)
        
        m = mean(X(idx==i,1));
        [~,~,idx_mean]=unique(round(abs(mu-m)));
        b=mu(idx_mean==1);
        
        if length(b) >1 
            r = fix(rand*length(b));
            f =b(r+1);
            z(i)=f;

           
            %% assign index mu 
            while  ismember(f,ind_mu) 
                    r = fix(rand*length(b));
                    array_r(i)= r;
                    f =b(r+1);
                  
            end
            
            ind_mu(i) = find(mu == f);
        else 
            z(i)=b;
            if ismember(find(mu == b),ind_mu)
             %%assing index mu 
             disp('here')
            else 
                ind_mu(i) = find(mu == b);
             end
            
        end
        
    end
    
    %now we know the order of the indexs 
    % class_mean(1) == (idx ==1)
    %ind_mu = find(mu == class_mean(1));
    
    
    %%work out the error 
    
    %num_c_pre = find(idx == find(ind_mu==2)); %% difference(num_c vs num_c_pre ) is error 
    %error = abs(length(num_c_pre)-length(idx_c)); 
    
    
    p= [2,3,4,1];
    error =0;
    for i=1:1:length(C)
        num_pre = find(idx == find(ind_mu==p(i)));
        error = error + abs(length(num_pre)-idx_real(i));
    end

    error_tot(ii) = error;
end

%%stats%%

avg_error = 1- (sum(error_tot)/length(error_tot))/length(X);
avg_error = num2str(avg_error);
disp('the classifation accury ');
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
title('syntic data');

%print the predication with K means 

figure;
plot(X(idx==find(ind_mu==1),1),X(idx==find(ind_mu==1),2),'r.','MarkerSize',12) %other
hold on
plot(X(idx==find(ind_mu==2),1),X(idx==find(ind_mu==2),2),'b.','MarkerSize',12) %cat
hold on
plot(X(idx==find(ind_mu==3),1),X(idx==find(ind_mu==3),2),'g.','MarkerSize',12) %dog
hold on
plot(X(idx==find(ind_mu==4),1),X(idx==find(ind_mu==4),2),'k.','MarkerSize',12) %mouse
plot(C(:,1),C(:,2),'mx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off