load('session1data');
dataset = session1dataset;

[NumRow,NumCol] = size(dataset);
index = randperm(NumRow);
dataset_shuffle = dataset(index, :);

% data의 95%는 training, 5%는 test
Train = dataset_shuffle(1:342,:);
Test = dataset_shuffle(343:360,:);

XTrain = Train.eegdata;
YTrain = categorical(Train.label);

%% data ploting
figure
plot(XTrain{1}')
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
ylim([0 60000])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")


%% LSTM layer

numFeatures = 62;
numHiddenUnits1 = 50;
numHiddenUnits2 = 40;
numClasses = 4;
layers = [ ...
    sequenceInputLayer(numFeatures)
    batchNormalizationLayer
    bilstmLayer(numHiddenUnits1,'OutputMode','sequence')
    reluLayer
    dropoutLayer(0.4)
    batchNormalizationLayer
    bilstmLayer(numHiddenUnits2,'OutputMode','last')
    reluLayer
    dropoutLayer(0.2)
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
    'Shuffle','once', ...
    'Verbose',0, ...
    'Plots','training-progress');

%     'InitialLearnRate',0.01, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',1, ...
%     'LearnRateDropFactor',0.0004, ...


net = trainNetwork(XTrain,YTrain,layers,options);


%% Test

XTest = Test.eegdata;
YTest = categorical(Test.label);

numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end

[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);

XTest = XTest(idx);
YTest = YTest(idx);

YPred = classify(net,XTest, ...
    'MiniBatchSize',minibatchsize, ...
    'SequenceLength','shortest');

acc = sum(YPred == YTest)./numel(YTest)

 
