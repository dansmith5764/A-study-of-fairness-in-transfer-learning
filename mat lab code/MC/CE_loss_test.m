clear 
close all 
clc


rng('default') % for reproducibility of demo
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2);
    randn(100,2)*0.75];
groupID = repelem(1:3,1,100)';   % known group ID for each point
figure()
gscatter(X(:,1), X(:,2), groupID, 'bgm', 'o')
title('Raw data')

k = 3;
[idx,C] = kmeans(X,k);
% Plot clusters
figure(); 
gscatter(X(:,1), X(:,2), idx, 'bgm', 'x')
title('Clustered data')


T = array2table(zeros(k,3),'VariableName',{'cluster','dominantGroup','percentSame'}); 
for i = 1:k
    counts = histcounts(groupID(idx==i),'BinMethod','integers','BinLimits',[1,k]);
    [maxCount, maxIdx] = max(counts);
    T.cluster(i) = i; 
    T.dominantGroup(i) = maxIdx;
    T.percentSame(i) = maxCount/sum(counts);
end
disp(T)