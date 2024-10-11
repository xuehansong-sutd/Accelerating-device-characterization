%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bayesian optimization of hyperparameters for NN
% For single junction PSC
%
% Authors: Tan Hu Quee, Zhao Xinhai, Erik Birgersson
% Data modified: 26 April 2024
%
% Parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data import
% Training data
input_data_train = load('Data_10k_rng1_v2\input.mat'); % input data (seen)
output_data_train = load('Data_10k_rng1_v2\output1'); % Output data (seen)
input_T = input_data_train.input;
output_T = output_data_train.output;

% Validation data
input_data_vali = load('Data_1k_rng1\input.mat'); % input data (unseen)
output_data_vali = load('Data_1k_rng1\output1_v2_Voc_iMPP_VMPP.mat'); % Output data (unseen)
input_V = input_data_vali.input;
output_V = output_data_vali.output;

% Test data
input_data_test = load('Data_1k_rng2\input.mat'); % input data (unseen)
output_data_test = load('Data_1k_rng2\output1_v2_Voc_iMPP_VMPP.mat'); % Output data (unseen)
input_t = input_data_test.input;
output_t = output_data_test.output;

%% Data processing
% Take log10 for certain parameters
% as to improve NN performance
input_T(:,[4:13,26:31]) = log10(input_T(:,[4:13,26:31]));
input_V(:,[4:13,26:31]) = log10(input_V(:,[4:13,26:31]));
input_t(:,[4:13,26:31]) = log10(input_t(:,[4:13,26:31]));

%% Data normalization
% we normalize the data between -1 and 1
% as to improve NN performance

i_T = size(input_T,1);
i_V = size(input_V,1);
i_t = size(input_t,1);

input_all = [input_T; input_V; input_t].';
output_all = [output_T; output_V; output_t].';

[input,PS_input]= mapminmax(input_all,-1,1);
[output,PS_output] = mapminmax(output_all,-1,1);
% output = normalize(output_all);

%% Colors for postprocessing
Green = [146/255,208/255,80/255];
Red = [1,0,0];
Blue = [0,0,1];

%% Hyperparameters in Bayesian optimziation
% We optimize number of hidden layers, size of layers, number of epochs,
% initial learning rate, momentum, and transfer functions for the hidden layers
% For more ways to optimize hyperparaters, see e.g. https://en.wikipedia.org/wiki/Hyperparameter_optimization
MaxObjBay = 1e4;     % Maximum number of objective iterations
MaxTimeBay = 6.7*24*60*60;   % Maximum time, seconds
NumSeed = 5e2; % Number of initial seed points

optimVars_4L = [optimizableVariable('Layer1Size',[2 100],'Type','integer')
    optimizableVariable('Layer2Size',[2 100],'Type','integer')
    optimizableVariable('Layer3Size',[2 100],'Type','integer')
    optimizableVariable('Layer4Size',[2 100],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('MinGradient',[1e-8 1e-6],'Transform','log')
    optimizableVariable('Momentum',[0.5 0.98])
    optimizableVariable('transferFcn1',{'tansig' 'radbas' 'logsig' 'purelin'},'Type','categorical')
    optimizableVariable('transferFcn2',{'tansig' 'radbas' 'logsig' 'purelin'},'Type','categorical')
    optimizableVariable('transferFcn3',{'tansig' 'radbas' 'logsig' 'purelin'},'Type','categorical')
    optimizableVariable('transferFcn4',{'tansig' 'radbas' 'logsig' 'purelin'},'Type','categorical')];