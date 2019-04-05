% Load one of the provided dataset, e.g.
load('./Datasets/heart.mat')
[num_examples num_features] = size(data);
M = 5; % number of iterations to create matrix Z
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Select the features in each Bootstrap sample %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Initialise the selection matrices Z
Z_LASSO = zeros(M,num_features);
Z_EN = zeros(M,num_features);
Z_RF = zeros(M,num_features);
Z_CMIM = zeros(M,num_features);

disp(sprintf('Bootstrap sample (out of %d): ',M)); fprintf('\b');
for it_index = 1:M
    
    %%% Bootstrap samples
    Indices_train = randsample(num_examples,num_examples,true);
    
    fprintf('\b'); disp(sprintf('%d,',it_index));
    train_data = data(Indices_train,:); train_labels = labels(Indices_train,1);
    
    
    %%%% Selecting  features using
    %%%%%%%%%%%%%%%%%
    %%%% LASSO %%%%%%
    %%%%%%%%%%%%%%%%%
    [ba fitinfoa] = lassoglm(train_data,double(train_labels==1),'binomial','CV',2,'Alpha',1);
    features_LASSO = find(ba(:,find(fitinfoa.Lambda == fitinfoa.Lambda1SE))~=0)';
    % For Deviance
    Deviance_LASSO_it( it_index) = fitinfoa.Deviance(fitinfoa.Lambda == fitinfoa.Lambda1SE);
    % For Stability Calculatios
    Z_LASSO( it_index, features_LASSO) = 1;
    
    %%%%%%%%%%%%%%%%%%%%%
    %%%% Elastic Net %%%% with \alpha = 0.50
    %%%%%%%%%%%%%%%%%%%%%
    alpha_EN = 0.50;
    [ba fitinfoa] = lassoglm(train_data,double(train_labels==1),'binomial','CV',2,'Alpha',alpha_EN);
    features_EN = find(ba(:,find(fitinfoa.Lambda == fitinfoa.Lambda1SE))~=0)';
    % For Deviance
    Deviance_EN_it( it_index) = fitinfoa.Deviance(fitinfoa.Lambda == fitinfoa.Lambda1SE);
    % For Stability Calculatios
    Z_EN( it_index, features_LASSO) = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%
    %%%% Random Forest %%% with 1000 trees to select top-k=5 features
    %%%%%%%%%%%%%%%%%%%%%%
    num_trees = 128; topK = 5;
    Mdl = TreeBagger(num_trees,train_data,double(train_labels==1),'Method','Classification','OOBVarImp', 'on');
    [~,features_RF] = sort(Mdl.OOBPermutedVarDeltaError,'descend');
    Z_RF( it_index, features_RF(1:topK)) = 1;
    
    %%%%%%%%%%%%%%
    %%%% CMIM  %%% to select top-k=5 features
    %%%%%%%%%%%%%% to estimate mutual information we descretize in 5 bins
    bins = 5;
    train_data_disc = disc_dataset_equalwidth( train_data, bins );
    features_CMIM = CMIM(train_data_disc, double(train_labels==1), topK) ;
    Z_CMIM( it_index, features_CMIM) = 1;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Estimate the stabilities using prior knowledge %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Prior Knowledge over Feature Redundancy
C = eye(num_features);
%%%% Assuming features 1,2 and 3,4 are correlated:
C(1,2) = 1; C(2,1) = 1;
C(3,4) = 1; C(4,2) = 1;

Effective_Stab_Red_LASSO = effectiveStabilityWithRedundancy(Z_LASSO, C);
Effective_Stab_Red_EN = effectiveStabilityWithRedundancy(Z_EN, C);
Effective_Stab_Red_RF = effectiveStabilityWithRedundancy(Z_RF, C);
Effective_Stab_Red_CMIM = effectiveStabilityWithRedundancy(Z_CMIM, C);
disp('Effective stabilities accounting for redundancy between features 1,2 and 3,4:')
disp(sprintf('LASSO = %0.3f, Elastic Net = %0.3f, Random Forest = %0.3f, CMIM = %0.3f',Effective_Stab_Red_LASSO,Effective_Stab_Red_EN,Effective_Stab_Red_RF,Effective_Stab_Red_CMIM))

