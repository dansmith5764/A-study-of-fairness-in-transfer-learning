function [accuracy_means, std_mean, total_cross_loss] = BVG_MC_dist_means(P,MC, N, iterations,size_interation, Class,  cov)
    m1 = 0;
    m2 = 0;
    m3 = 1;
    m4 = 1 ;

for j=1:1:iterations
  
    mu_tot{j} = {[0 0] [m3 m4]};
    m3 = m3-size_interation;
    m4 = m4-size_interation;
end 

%% loop for differnt opperating contionous 
for iii =1:1:length(mu_tot)

    mu = mu_tot{iii};
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
        cross_loss(ii) = crossentropy(idx_real,idx);
    end

    %%histcounts loss
    accuracy_iterations(iii)  =  sum(totat_accuracy)/MC;
    std_mean(iii) = std(accuracy_iterations);
    
    %%% KDL loss 
    KDL_total(iii)= sum(KLD)/MC;
    total_cross_loss(iii) = sum(cross_loss)/MC;
end
%accuracy_means  =  sum(totat_accuracy)/MC;
accuracy_means= accuracy_iterations;



% % plot the average value from all the simulations 
%     figure;
%     hold on
%     title('monty carlo simulation');
%     plot( 1:length(mu_tot),accuracy_iterations);
%     plot( 1:length(mu_tot),accuracy_iterations+std_iterations, 'r')
%     plot( 1:length(mu_tot),accuracy_iterations-std_iterations, 'r')
%     
%     %plot(1:length(mu_tot), KDL_total, 'r');
%     hold off
end




