load('data');

[NumRow,NumCol] = size(dataset);
index = randperm(NumRow);
dataset_shuffle = dataset(index, :);

% data의 95%는 training, 5%는 test
Train = dataset_shuffle(1:1026,:);
Test = dataset_shuffle(1027:1080,:);

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

inputSize = 62;
numHiddenUnits = 20;
numClasses = 4;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]


maxEpochs = 20;
miniBatchSize = 9;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

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
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest)
% acc = 24%나옴

