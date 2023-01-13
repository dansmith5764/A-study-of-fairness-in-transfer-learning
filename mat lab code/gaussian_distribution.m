function X = gaussian_distribution(mu,mu_y,sigma, Class_matrix)

Num_c = find(Class_matrix(:,2) == 1); %% cats are circles and 1's
Num_d = find(Class_matrix(:,2) == 2);
Num_m = find(Class_matrix(:,2) == 3);
Num_o = find(Class_matrix(:,2) == 0);
L = [length(Num_c), length(Num_d), length(Num_m), length(Num_o) ];


a = ones(length(Num_c),1);
b= 2*ones(length(Num_d),1);
c= 3*ones(length(Num_m),1);
d = zeros(length(Num_o),1);

%% generate the points 
cat_x  = repmat(mu(2),L(1),1)+ randn(L(1),4)*sigma(:,1);
cat_y = repmat(mu_y(2),L(1),1)+ randn(L(1),4)*sigma(:,1);

dog_x  = repmat(mu(3),L(2),1)+ randn(L(2),4)*sigma(:,2);
dog_y = repmat(mu_y(3),L(2),1)+ randn(L(2),4)*sigma(:,2);

mouse_x  = repmat(mu(4),L(3),1)+ randn(L(3),4)*sigma(:,3);
mouse_y = repmat(mu_y(4),L(3),1)+ randn(L(3),4)*sigma(:,3);

other_x  = repmat(mu(1),L(4),1)+ randn(L(4),4)*sigma(:,4);
other_y = repmat(mu_y(1),L(4),1)+ randn(L(4),4)*sigma(:,4);




cat = [cat_x(1:end,1), cat_y(1:end,1),  a];
dog = [dog_x(1:end,1), dog_y(1:end,1), b ];
mouse = [mouse_x(1:end,1), mouse_y(1:end,1), c];
other = [other_x(1:end,1), other_y(1:end,1), d] ;


X= [cat ; dog; mouse; other];



end