function stability = effectiveStabilityWithRedundancy(Z, C)
% Summary
%    Calculates the stability estimator \hat{\Phi}}_{\mathbb{G}(\mathcal{Z})
%    Input: 
%       - Z: a binary matrix of size M*d where each row represents a feature s
%       - C: a matrix d*d that captures which pairs of features are correlated
%      
%    Output:
%       - stability: the stability estimate taking into account groups of features
%%% Numerator
effective_numerator = sum(sum(C.*((cov(Z)))));

%%% Denominator
p_f_element = mean(sum(Z,2))/size(Z,2); 
p_f = repmat(p_f_element,1,size(Z,2));
 
p_ff_element =  mean(sum(Z,2).*(sum(Z,2)-1) )./(size(Z,2)*(size(Z,2)-1));
p_ff = repmat(p_ff_element,size(Z,2),size(Z,2));
p_ff(1:(size(Z,2)+1):end) = p_f_element;

covariance_matrix = p_ff -  p_f' * p_f;
effective_denominator = sum(sum(C.*(((covariance_matrix)))));
 
%%% Effective Stability
stability = 1- effective_numerator/(effective_denominator);

