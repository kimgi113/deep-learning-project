load('psddataset_gamma.mat');


%dataset shuffle
[NumRow,NumCol] = size(dataset);
index = randperm(NumRow);
dataset_shuffle = dataset(index, :);

%ID 구분
ID = 6;
X = dataset_shuffle.ID;
Y = find(X==ID);
ID_dataset_shuffle = dataset_shuffle(Y,:);

%Label 정리
X = ID_dataset_shuffle.label;
Y = find(X==0);
label0 = ID_dataset_shuffle(Y,:);
X = ID_dataset_shuffle.label;
Y = find(X==1);
label1 = ID_dataset_shuffle(Y,:);
X = ID_dataset_shuffle.label;
Y = find(X==2);
label2 = ID_dataset_shuffle(Y,:);
X = ID_dataset_shuffle.label;
Y = find(X==3);
label3 = ID_dataset_shuffle(Y,:);

%Train set, Validation set에 분배
datalength = height(ID_dataset_shuffle);
Train = vertcat(label0(1:16,:), label1(1:16,:), label2(1:16,:), label3(1:16,:));
Valid = vertcat(label0(16:17,:), label1(16:17,:), label2(16:17,:), label3(16:17,:));
[NumRow,NumCol] = size(Train);
index = randperm(NumRow);
Train = Train(index, :);
[NumRow,NumCol] = size(Valid);
index = randperm(NumRow);
Valid = Valid(index, :);



% % data의 90%는 training, 5%는 validation, 5%는 test
% datalength = height(dataset_shuffle);
% Train = dataset_shuffle(1:datalength*0.9,:);
% Valid = dataset_shuffle(datalength*0.9+1:datalength*0.95,:);
% Test = dataset_shuffle(datalength*0.95+1:datalength,:);

XTrain = Train.eegdata;
YTrain = categorical(Train.label);
XValid = Valid.eegdata;
YValid = categorical(Valid.label);

%% data ploting
figure
plot(XTrain{2}')
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

% %% Normalization
% mu = mean([XTrain{:}],2);
% sig = std([XTrain{:}],0,2);
% 
% for i = 1:numel(XTrain)
%     XTrain{i} = (XTrain{i} - mu) ./ sig;
% end
% 
% 
% mu = mean([XValid{:}],2);
% sig = std([XValid{:}],0,2);
% 
% for i = 1:numel(XValid)
%     XValid{i} = (XValid{i} - mu) ./ sig;
% end

%% LSTM layer

[inputSize, Times] = size(XTrain{1});
numHiddenUnits = 500;
numClasses = 4;

layers = [ ...
    sequenceInputLayer(inputSize,"Normalization","zscore","NormalizationDimension","channel")
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%       
%    dropoutLayer(0.5,"Name","dropout")
%    reluLayer("Name","relu")

minibatchsize = 5;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',300, ...
    'MiniBatchSize',minibatchsize, ...
    'SequenceLength','shortest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'L2Regularization', 0.01, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency',20, ...
    'ValidationPatience',Inf, ...
    'Plots','training-progress');
%    'L2Regularization', 0.001, ...


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

 
