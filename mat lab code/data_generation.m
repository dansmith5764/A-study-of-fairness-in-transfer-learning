function X = data_generation(N, mu, cov, P)

counter=0;
%%assign 
X= zeros(N,3);
for i=1:1:N
    x = rand();
    L(i)= x;
    %% others
    if x <= P(1)
        counter  = counter +1 ;
        X(i,:) = [repmat(mu{1},1) + randn(1,length(cov{1}))*cov{1} 1];
      
    %% cat
    elseif    P(1) <= x && x <= P(1)+P(2)
        counter  = counter +1 ;
        X(i,:) = [repmat(mu{2},1) + randn(1,length(cov{2}))*cov{2} 2 ];
       
    %% dog
    elseif    P(1)+P(2) <= x && x <= P(1)+P(2)+P(3)
        counter  = counter +1 ;
        X(i,:) = [repmat(mu{3},1) + randn(1,length(cov{3}))*cov{3} 3];
    else %mice
        counter  = counter +1 ;
        X(i,:) = [repmat(mu{4},1) + randn(1,length(cov{4}))*cov{4} 4];
    end 

end


end

%% all values between 0-1
%% 25% each 
%% 4 classifers 