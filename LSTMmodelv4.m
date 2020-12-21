load('psddataset_gamma_prefrontal');

[NumRow,NumCol] = size(dataset);
index = randperm(NumRow);
dataset_shuffle = dataset(index, :);

% data의 90%는 training, 5%는 validation, 5%는 test
datalength = height(dataset);
Train = dataset_shuffle(1:datalength*0.9,:);
Valid = dataset_shuffle(datalength*0.9+1:datalength*0.95,:);
Test = dataset_shuffle(datalength*0.95+1:datalength,:);

XTrain = Train.eegdata;
YTrain = categorical(Train.label);
XValid = Valid.eegdata;
YValid = categorical(Valid.label);

%% data ploting
figure
plot(XTrain{700}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')


%% data sort
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);

figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")


%% LSTM layer

inputSize = 5;
numHiddenUnits = 50;
numClasses = 4;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

minibatchsize = 9;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',minibatchsize, ...
    'SequenceLength','shortest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency',20, ...
    'ValidationPatience',Inf, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

    

% %% Test
% 
% XTest = Test.eegdata;
% YTest = categorical(Test.label);
% 
% numObservationsTest = numel(XTest);
% for i=1:numObservationsTest
%     sequence = XTest{i};
%     sequenceLengthsTest(i) = size(sequence,2);
% end
% 
% [sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
% 
% XTest = XTest(idx);
% YTest = YTest(idx);
% 
% YPred = classify(net,XTest, ...
%     'MiniBatchSize',minibatchsize, ...
%     'SequenceLength','shortest');
% 
% acc = sum(YPred == YTest)./numel(YTest)
% 
%  
