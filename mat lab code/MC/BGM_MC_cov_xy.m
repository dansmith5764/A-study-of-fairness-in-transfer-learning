function [accuracy_cov,std_cov] = BGM_MC_cov_xy(P,MC, N, iterations, Class,  size_interation,mu) 
% assign mean to the each class
    c1 = 0;
    c2 = 0;

for j=1:1:iterations 
  
    cov_tot{j} = {[1 0; 0 1]  [1 c1; c2 1]};
    c1 = c1+size_interation;
    c2 = c2+size_interation;
end 


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
        cross_loss = crossentropy(idx_real,idx);
    end

    %%histcounts loss
    accuracy_iterations(iii)  =  sum(totat_accuracy)/MC;
    std_cov = std(totat_accuracy);
    %%% KDL loss 
    KDL_total(iii)= sum(KLD)/MC;
    total_cross_loss = sum(cross_loss)/MC;
end

%accuracy_cov  =  sum(totat_accuracy)/MC;

accuracy_cov = accuracy_iterations;


end