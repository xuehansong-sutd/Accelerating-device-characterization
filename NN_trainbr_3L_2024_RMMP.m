%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Training of neural networks (Rec at MPP)
%
% Author: Zhao Xinhai
% Last modified: 2024.10.11
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
clear all
close all
clc
format compact
rng default

%% Data import
% Training data
input_data_train = load('Data_100k_v2\input.mat'); % input data (seen)
output_data_train = load('Data_100k_v2\output2.mat'); % Output data (seen)
input_T = input_data_train.input;
output_T = output_data_train.output;

% Validation data
input_data_vali = load('Data_10k_rng1_v2\input.mat'); % input data (unseen)
output_data_vali = load('Data_10k_rng1_v2\output2.mat'); % Output data (unseen)
input_V = input_data_vali.input;
output_V = output_data_vali.output;

% Test data
input_data_test = load('Data_10k_rng2_v2\input.mat'); % input data (unseen)
output_data_test = load('Data_10k_rng2_v2\output2.mat'); % Output data (unseen)
input_t = input_data_test.input;
output_t = output_data_test.output;

%% Data processing
% Take log10 for certain parameters
% as to improve NN performance
input_T(:,[4:13,26:31]) = log10(input_T(:,[4:13,26:31]));
output_T = log10(output_T);
input_V(:,[4:13,26:31]) = log10(input_V(:,[4:13,26:31]));
output_V = log10(output_V);
input_t(:,[4:13,26:31]) = log10(input_t(:,[4:13,26:31]));
output_t = log10(output_t);

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

%% NN
% Training function for NN
% https://www.mathworks.com/help/deeplearning/ref/trainbr.html
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.
HLSize = [61 39 56];
net = fitnet(HLSize,trainFcn);
% Division of Data for Training, Validation, Testing
net.divideFcn = 'divideind';
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainInd = 1:i_T;
net.divideParam.valInd = i_T+1:i_T+i_V;
net.divideParam.testInd = i_T+i_V+1:i_T+i_V+i_t;
net.performFcn = 'mse'; % Performance function
net.performParam.normalization = 'standard';
% Other hyperparameters
net.trainParam.max_fail = 10;
net.trainParam.epochs = 1000;   % Number of epochs
net.trainParam.lr = 0.0135; % Initial learning rate
net.trainParam.min_grad = 7.08e-8;
net.trainParam.mc = 0.727;  % Momentum
% Transfer functions
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'tansig';
% Train the NN
[net,valError,tr] = train(net,input,output,'useParallel','no');
save('NN_3L_Perfold_100k_RMPP','net','valError','tr','PS_input','PS_output');