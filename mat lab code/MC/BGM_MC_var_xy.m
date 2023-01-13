function [accuracy_var, std_var] = BGM_MC_var_xy(P,MC, N, iterations,size_interation, Class,  mu)



% assign mean to the each class
    c1 = 1.02;
    c2 = 1.02;

for j=1:1:iterations
  
    cov_tot{j} = {[1 0; 0 1]  [c1 0; 0 c2]};
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
    end

    %%histcounts loss
    accuracy_iterations(iii)  =  sum(totat_accuracy)/MC;
    std_var(iii) = std(totat_accuracy);
    %%% KDL loss 
    KDL_total(iii)= sum(KLD)/MC;
end
%accuracy_var  =  sum(totat_accuracy)/MC;

accuracy_var = accuracy_iterations;


% plot the average value from all the simulations 
%     figure;
%     hold on
%     title('monty carlo simulation');
%     plot( 1:length(cov_tot),accuracy_iterations);
%     plot(1:length(mu_tot), KDL_total, 'r');
%     hold off

end


