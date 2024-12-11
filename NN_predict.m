%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NN prediction
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

rng default
load('FinalNN\NN_Perf_final.mat')

%% Input parameters
LH = 25; LP = 300; LE = 25; G = 260/1.602e-19/(LP*1e-9);
muHh = 1e-6; muPh = 5.5e-4; muPe = 5.5e-4; muEe = 1e-4;
NvH = 1e25; NcH = 1e25; NvE = 1e25; NcE = 1e25; NvP = 1e25; NcP = 1e25;
chiHh = 5.3; chiHe = 2.1; chiPh = 5.5; chiPe = 4.0; chiEh = 6; chiEe = 4.2;
Wlm = 4.3; Whm = 4.9; epsH = 3; epsP = 6; epsE = 3;
Aug = 1e-40; Brad = 1e-17; tauh = 1e-5; taue = 1e-5; vIV = 1e-25; vV = 1e-25;

X_P = [LH,LP,LE,muHh,muPh,muPe,muEe,NvH,NcH,NvE,NcE,NvP,NcP,...
    chiHh,chiHe,chiPh,chiPe,chiEh,chiEe,Wlm,Whm,epsH,epsP,epsE,G,Aug,Brad,tauh,taue,vIV,vV];
X_P([4:13,26:31]) = log10(X_P([4:13,26:31]));

input = mapminmax('apply',X_P.',PS_input);

%% Predict
tic
output = net(input);
toc

Perf = mapminmax('reverse',output,PS_output);
Voc = Perf(1)
FF = Perf(2)
PCE = Perf(3)