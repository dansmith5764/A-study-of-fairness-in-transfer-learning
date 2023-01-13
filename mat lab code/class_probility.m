function X = class_probility(size, prob)
C = [0,1,2,3];
X = zeros(size,2) ;
%%assign 
counter  = 0;
for i=1:1:size
    x = rand();
    X(i)= x;
    %% cat 
    if    x <= prob(1)
        counter  = counter +1 ;
        X(i,2) = C(2);
    end
    %% dog
    if    prob(1) <= x && x <= prob(1)+prob(2)
        counter  = counter +1 ;
        X(i,2) = C(3);
    end
    %% mouse
    if    prob(1)+prob(2) <= x && x <= prob(1)+prob(2)+prob(3)
        counter  = counter +1 ;
        X(i,2) = C(4);
    end
end

end

%% all values between 0-1
%% 25% each 
%% 4 classifers 