function [y, y_k ] = PDF(Class, idx, idx_real, mu, cov,N,X)

%Evaluate the pdf of the distribution at the points in X.
for i=1:1:length(Class)

    p_k = length(X(idx==i,1))/N;
    Y_k = [ X(idx==i,1) X(idx==i,2)];
    mu_x  =  mean(X(idx==i,1));
    mu_y = mean(X(idx==i,2)); 
    mu1 = {[mu_x mu_y]};
    y_k{i}= p_k*mvnpdf(Y_k,mu1{1}, cov{i});
    

    p = length(X(idx_real==i,1))/N;
    Y = [ X(idx_real==i,1) X(idx_real==i,2)];
    y{i}= p*mvnpdf(Y,mu{i}, cov{i});
    
end
