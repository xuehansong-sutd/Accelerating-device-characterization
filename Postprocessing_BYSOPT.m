%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bayesian optimization of hyperparameters for NN
% For single junction PSC
%
% Authors: Tan Hu Quee, Zhao Xinhai, Erik Birgersson
% Data modified: 19 April 2023
%
% Postprocessing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load saved net
% load('OptimizedNN_3L_RVoc.mat');

%% Number of parameters in NN
Weight_and_Bias = getwb(net);
numParams = numel(Weight_and_Bias);
% Display the total number of parameters
disp(['Total number of parameters: ' num2str(numParams)]);

%% Delete unnecessary mat file
% Delete all the generated NNs that we do not need.
% Please rename/save the target NNs before delete
delete('temp_*.mat'); % uncommment this lint to delete temp files

%% Unseen data
input_u = readtable('Data_10k\LHS_parameters_m.txt'); % input data (unseen)
output_u = readtable('Data_10k\CellPerformance.txt'); % Output data (unseen)
input_u = table2array(input_u);
output_u = table2array(output_u);
output_u = output_u(:,10:14);
input_u(:,[4:13,26:31]) = log10(input_u(:,[4:13,26:31]));
output_u = log10(output_u);
% index_r = randi(10000,100,1);
% input_u = input_u(index_r,:);
% output_u = output_u(index_r,:);
input_u = mapminmax('apply',input_u.',PS_input);
output_u = mapminmax('apply',output_u.',PS_output);

%% NN performance
YPredicted = net(input);
valError = perform(net,output,YPredicted);
YPredicted_u = net(input_u);
valError_u = perform(net,output_u,YPredicted_u);

% Visualization of NN perfomance
% Xlabelnames = {'{\it{i_{sc}}} (predicted)', '{\it{V_{oc}}} (predicted)', 'FF (predicted)', 'PCE (predicted)',...
%     'Rrad (MPP) (predicted)', 'Rsrh (MPP) (predicted)', 'Raug (MPP) (predicted)','RII (MPP) (predicted)','RIII (MPP) (predicted)',...
%     'Rrad (OC) (predicted)', 'Rsrh (OC) (predicted)', 'Raug (OC) (predicted)','RII (OC) (predicted)','RIII (OC) (predicted)'};
% Ylabelnames = {'{\it{i_{sc}}} (actual)', '{\it{V_{oc}}} (actual)', 'FF (actual)', 'PCE (actual)',...
%     'Rrad (MPP) (actual)', 'Rsrh (MPP) (actual)', 'Raug (MPP) (actual)','RII (MPP) (actual)','RIII (MPP) (actual)',...
%     'Rrad (OC) (actual)', 'Rsrh (OC) (actual)', 'Raug (OC) (actual)','RII (OC) (actual)','RIII (OC) (actual)'};
Xlabelnames = {'Rrad (OC) (predicted)', 'Rsrh (OC) (predicted)', 'Raug (OC) (predicted)','RII (OC) (predicted)','RIII (OC) (predicted)'};
Ylabelnames = {'Rrad (OC) (actual)', 'Rsrh (OC) (actual)', 'Raug (OC) (actual)','RII (OC) (actual)','RIII (OC) (actual)'};

figure(1)
tiledlayout(3,2)
for i = 1:size(output,1)
    nexttile
    % True vs predicted values for seen dataset
    plot(YPredicted(i,:),output(i,:),'b+')
    hold on
    
    % True vs predicted values for unseen data set
    plot(YPredicted_u(i,:),output_u(i,:).','ro')
    axis square
    hold off

    xlabel(Xlabelnames{i},'fontsize',14,'fontname','Times New Roman');
    ylabel(Ylabelnames{i},'fontsize',14,'fontname','Times New Roman');
end

%% Performance metrics
trainOut = YPredicted(:,tr.trainInd);
valOut = YPredicted(:,tr.valInd);
testOut = YPredicted(:,tr.testInd);
trainTarg = output(:,tr.trainInd);
valTarg = output(:,tr.valInd);
testTarg = output(:,tr.testInd);
figure(2)
plotperf(tr)
figure(3)
plotregression(trainTarg, trainOut, 'Train', valTarg, valOut, 'Validation', testTarg, testOut, 'Testing')

%% NN performance (raw data, without nomalization)
YPredicted0 = mapminmax('reverse',YPredicted,PS_output);
YPredicted0_u = mapminmax('reverse',YPredicted_u,PS_output);
output0 = mapminmax('reverse',output,PS_output);
output0_u = mapminmax('reverse',output_u,PS_output);

figure(4)
tiledlayout(3,2)
for i = 1:size(output0,1)
    nexttile
    % True vs predicted values for seen dataset
    plot(YPredicted0(i,1:65844),output0(i,1:65844),'b+')
    hold on
    
    % True vs predicted values for unseen data set
    plot(YPredicted0_u(i,:),output0_u(i,:).','ro')
    hold off

    axis square

    xlabel(Xlabelnames{i},'fontsize',14,'fontname','Times New Roman');
    ylabel(Ylabelnames{i},'fontsize',14,'fontname','Times New Roman');
end