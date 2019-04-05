function selectedFeatures = CMIM(X,Y, topK)
% Summary
%    Return indices of the selected features using CMIM
%    Input: 
%       - X: the feature matrix n*d
%       - y: the label vector n*1
%       - topK: the number of selected features
%
%    Output:
%       - selectedFeatures: the selected set

numFeatures = size(X,2);

mi_score = zeros(1,numFeatures);

for index_feature = 1:numFeatures
    index_feature;
    score_per_feature_uni(index_feature) = mi(X(:,index_feature),Y);
end
[val_max,selectedFeatures(1)]= max(score_per_feature_uni);
mi_score(selectedFeatures(1)) = val_max;
not_selected_features = setdiff(1:numFeatures,selectedFeatures);

%%% Efficient implementation of the second step, at this point I will store
%%% the score of each feature. Whenever I select a feature I put NaN score
score_per_feature = ones(1,numFeatures)*NaN;
score_per_feature(selectedFeatures(1)) = NaN;
count = 2;
while count<=topK
    
    for index_feature_ns = 1:length(not_selected_features)        
        score_per_feature(not_selected_features(index_feature_ns)) = min(score_per_feature(not_selected_features(index_feature_ns)),cmi(X(:,not_selected_features(index_feature_ns)), Y, X(:, selectedFeatures(count-1))));        
    end       
    
    [val_max,selectedFeatures(count)]= nanmax(score_per_feature);
    
    
    score_per_feature(selectedFeatures(count)) = NaN;
    not_selected_features = setdiff(1:numFeatures,selectedFeatures);
    count = count+1;
end


