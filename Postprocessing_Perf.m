%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Postprocessing to evaluate NN performance
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
load('FinalNN\NN_Perf_final.mat')

%% Define colors
blue = [0.00 0.00 0.55];
red = [0.65 0.16 0.16];
green = [0.00 0.39 0.00];

%% Data import
% Training data
input_data_train = load('Data_100k_v2\input.mat'); % input data (seen)
output_data_train = load('Data_100k_v2\output.mat'); % Output data (seen)
input_T = input_data_train.input;
output_T = output_data_train.output;

% Validation data
input_data_vali = load('Data_10k_rng1_v2\input.mat'); % input data (unseen)
output_data_vali = load('Data_10k_rng1_v2\output.mat'); % Output data (unseen)
input_V = input_data_vali.input;
output_V = output_data_vali.output;

% Test data 1
input_data_test1 = load('Data_10k_rng2_v2\input.mat'); % input data (unseen)
output_data_test1 = load('Data_10k_rng2_v2\output.mat'); % Output data (unseen)
input_t1 = input_data_test1.input;
output_t1 = output_data_test1.output;

% Test data 2
input_data_test2 = load('Data_10k_rng3_v2\input.mat'); % input data (unseen)
output_data_test2 = load('Data_10k_rng3_v2\output.mat'); % Output data (unseen)
input_t2 = input_data_test2.input;
output_t2 = output_data_test2.output;

%% Data processing
input_T(:,[4:13,26:31]) = log10(input_T(:,[4:13,26:31]));
input_V(:,[4:13,26:31]) = log10(input_V(:,[4:13,26:31]));
input_t1(:,[4:13,26:31]) = log10(input_t1(:,[4:13,26:31]));
input_t2(:,[4:13,26:31]) = log10(input_t2(:,[4:13,26:31]));

output_T = log10(output_T);
output_V = log10(output_V);
output_t1 = log10(output_t1);
output_t2= log10(output_t2);

%% Data normalization
input_Tn = mapminmax('apply',input_T.',PS_input);
output_Tn = mapminmax('apply',output_T.',PS_output);

input_Vn = mapminmax('apply',input_V.',PS_input);
output_Vn = mapminmax('apply',output_V.',PS_output);

input_t1n = mapminmax('apply',input_t1.',PS_input);
output_t1n = mapminmax('apply',output_t1.',PS_output);

input_t2n = mapminmax('apply',input_t2.',PS_input);
output_t2n = mapminmax('apply',output_t2.',PS_output);

%% Postprocessing
Y_Tn = net(input_Tn);
Y_T = mapminmax('reverse',Y_Tn,PS_output);

Y_Vn = net(input_Vn);
Y_V = mapminmax('reverse',Y_Vn,PS_output);

Y_t1n = net(input_t1n);
Y_t1 = mapminmax('reverse',Y_t1n,PS_output);

Y_t2n = net(input_t2n);
Y_t2 = mapminmax('reverse',Y_t2n,PS_output);

% Evaluate the model
Y_mse_T = mean((Y_Tn - output_Tn).^2,"all"); % Mean Squared Error
Y_mse_V = mean((Y_Vn - output_Vn).^2,"all"); % Mean Squared Error
Y_mse_t1 = mean((Y_t1n - output_t1n).^2,"all"); % Mean Squared Error
Y_mse_t2 = mean((Y_t2n - output_t2n).^2,"all"); % Mean Squared Error

Y_abse_T = abs(Y_T.' - output_T); 
Y_abse_V = abs(Y_V.' - output_V); 
Y_abse_t1 = abs(Y_t1.' - output_t1); 
Y_abse_t2 = abs(Y_t2.' - output_t2); 

Y_r2_T = 1 - sum((Y_Tn - output_Tn).^2)/sum((output_Tn - mean(output_Tn)).^2);
Y_r2_V = 1 - sum((Y_Vn - output_Vn).^2)/sum((output_Vn - mean(output_Vn)).^2);
Y_r2_t1 = 1 - sum((Y_t1n - output_t1n).^2)/sum((output_t1n - mean(output_t1n)).^2);
Y_r2_t2 = 1 - sum((Y_t2n - output_t2n).^2)/sum((output_t2n - mean(output_t2n)).^2);