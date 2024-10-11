 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bayesian optimization of hyperparameters for NN
% For single junction PSC
%
% Authors: Tan Hu Quee, Zhao Xinhai, Erik Birgersson
% Data modified: 19 April 2023
%
% NN training, 3 hidden layer
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Bayesian Optimization
ObjFcn = makeObjFcn(input,output,i_T,i_V,i_t);
BayesObject = bayesopt(ObjFcn,optimVars_3L,...
    'OutputFcn',@stopfun,...
    'MaxObjectiveEvaluations',MaxObjBay,...
    'MaxTime',MaxTimeBay,...
    'IsObjectiveDeterministic',true,...
    'UseParallel',false,...
    'NumSeedPoints',NumSeed);

% Print the final table with optimized hyperparameters to the screen
bestIdx = BayesObject.IndexOfMinimumTrace(end);
Table = BayesObject.XTrace(bestIdx,:)

% Save the optimized hyperparameters
OptimizedVars = BayesObject.XAtMinEstimatedObjective;
save('OptimizedVars_3L','OptimizedVars');

% Save the best NN.
fileName = BayesObject.UserDataTrace{bestIdx};
load(fileName);
save('OptimizedNN_3L','net','valError','tr','PS_input','PS_output');

%% Objective Function For Optimisation
function ObjFcn = makeObjFcn(X,Y,i_T,i_V,i_t)
    ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
%    function valError = valErrorFun(optVars)
        % Set the network parameters
        HLSize = [optVars.Layer1Size optVars.Layer2Size optVars.Layer3Size];
        net = fitnet(HLSize,'trainbr'); %trainbr
        % Division of Data for Training, Validation, Testing
        net.divideFcn = 'divideind';
        net.divideMode = 'sample';
        net.divideParam.trainInd = 1:i_T;
        net.divideParam.valInd = i_T+1:i_T+i_V;
        net.divideParam.testInd = i_T+i_V+1:i_T+i_V+i_t;
        net.performFcn = 'mse';     % Performance function
        net.performParam.normalization = 'standard';
        net.trainParam.max_fail = 10;
        net.trainParam.lr = optVars.InitialLearnRate;   % Initial learning rate
        net.trainParam.mc = optVars.Momentum;           % Momentum
        net.trainParam.epochs = 1000;        % Number of epochs
        net.trainParam.min_grad = optVars.MinGradient;  % Set the minimum gradient
        tF = {char(optVars.transferFcn1);char(optVars.transferFcn2);char(optVars.transferFcn3)};
        net.layers{1}.transferFcn = tF{1};
        net.layers{2}.transferFcn = tF{2};
        net.layers{3}.transferFcn = tF{3};
        net.layers{4}.transferFcn = 'purelin';  % Output layer

        % Train the Network
        [net,tr] = train(net,X,Y,'useParallel','no');
        valError = tr.best_perf; % best performance error achieved during training
        
        % comment out the following lines if saving of NNs during training is not wanted
        % Save the NNs
        fileName = "temp_" + num2str(valError) + ".mat";
        save(fileName,'net','valError','tr');
        cons = [];
    end
end

%% Output halt function 
function stop = stopfun(results,state)
    stop = false;
    switch state
        case 'iteration'
            if results.MinObjective < 1e-6
                stop = true;
            end
    end
end