%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bayesian optimization of hyperparameters for NN
% For single junction PSC
%
% Authors: Tan Hu Quee, Zhao Xinhai, Erik Birgersson
% Data modified: 19 April 2023
%
% Main
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
% Clear variables
clear all
close all
clc
format compact
rng default

%% Parameters
Parameters

%% Optimize hyperparameters for NN (with Bayesian optimization)
OptimizationNN_3Layer % 4 hidden layer NN

%% Postprocessing for NN performance
Postprocessing_BYSOPT