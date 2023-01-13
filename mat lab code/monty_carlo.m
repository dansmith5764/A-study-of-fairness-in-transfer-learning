clear 
close all 
clc


% generate random data with fixed mean and covarinace 


mu = [0 5 10 20 ];
sigma = [1 0.33 0.33 0.33 ; 0.33 1 0.33 0.33 ; 0.33 0.33 1 0.33; 0.33 0.33 0.33 1];

%% matrix factorization 
%R  = chol(sigma);

cat = zeros(100,3);
dog = zeros(100,3);


z_x = repmat(mu,100,1)+ randn(100,4)*sigma;
z_y = repmat(mu,100,1)+ randn(100,4)*sigma;


a= zeros(100,1);
b = 2*ones(100,1);
c= 3*ones(100,1);
d =ones(100,1);
cat = [z_x(1:end,1), z_y(1:end,1),  d];
dog = [z_x(1:end,2), z_y(1:end,2), b ];
mice = [z_x(1:end,3), z_y(1:end,3), c];
other = [z_x(1:end,4), z_y(1:end,4), a] ;


X= [cat ; dog; mice; other];

idx_c = find(X(:,3) == 1); %% cats are circles and 1's
idx_d = find(X(:,3) == 2); %% dogs are starts and 2's
idx_m = find(X(:,3) == 3); %% mice are squares and 3's
idx_o = find(X(:,3) == 0); %% other are plus and 0's

figure; 
plot(X(idx_c,1), X(idx_c,2), 'k o');
hold on
plot(X(idx_d,1), X(idx_d,2), '*c');
hold on
plot(X(idx_m,1), X(idx_m,2), 'b s');
hold on
plot(X(idx_o,1), X(idx_o,2), 'r +');
title('syntic data');





%% k-means magic
opts = statset('Display','final');
[idx,C] = kmeans(X,4,'Distance','cityblock',...
    'Replicates',5,'Options',opts);

%%work out the error 
%% first cats 
%%  all points from the idx =
% X(idx==1,1),X(idx==1,2)

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
hold on
plot(X(idx==4,1),X(idx==4,2),'k.','MarkerSize',12)
plot(C(:,1),C(:,2),'mx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off
