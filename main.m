%% Keyword Spotting System using low-complexity Feature Extraction and quantized LSTM
% This code present the feature extraction, the training and the
% quantization methods used with a 64-hidden unit LSTM.

url = 'https://ssd.mathworks.com/supportfiles/audio/google_speech.zip';
downloadFolder = tempdir;
dataFolder = fullfile(downloadFolder,'google_speech');

if ~exist(dataFolder,'dir')
    disp('Downloading data set (1.4 GB) ...')
    unzip(url,downloadFolder)
end

%% Feature Extraction

% Creation of Train, Validation and Test AudioDatastores.
datafolder = '../matlab_lstm/google_speech';
ads = audioDatastore(fullfile(datafolder,'train'), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

commands = categorical(["zero","one","two","three","four","five","six","seven","eight","nine"]);
isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsTrain = subset(ads,isCommand|isUnknown);
countEachLabel(adsTrain)

ads = audioDatastore(fullfile(datafolder,'validation'), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsValidation = subset(ads,isCommand|isUnknown);
countEachLabel(adsValidation)

ads = audioDatastore(fullfile(datafolder,'test'), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsTest = subset(ads,isCommand|isUnknown);
countEachLabel(adsTest)

%% Creation of the feature Extractor Instances
feOptions = featureExtractionOptions("fs",16e3,...
                                    "fmin",50,...
                                    "fmax",5000,...
                                    "numberBands",16,...
                                    "q",1.3,...
                                    "order",3,...
                                    "frameLength",25e-3,...
                                    "hopLength",10e-3,...
                                    "scale","Log",...
                                    "segmentDuration",1,...
                                    "padding",true,...
                                    "parallel",true);
                                
afe = FeatureExtractor(feOptions);

% Spectrogram

XTrain = afe.extract(adsTrain);
XValidation = afe.extract(adsValidation);
XTest = afe.extract(adsTest);

%% Modify Labels accordingly 

YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);
YTest = removecats(adsTest.Labels);

%% Display Spectrograms
specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))

    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
end

%% Add Background noise

adsBkg = audioDatastore(fullfile(datafolder, 'background'))
numBkgClips = 4000;
volumeRange = log10([1e-4,1]);

numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,'single');
bkgAll = readall(adsBkg);
ind = 1;

for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)

        x = bkg(idxStart(j):idxEnd(j))*gain(j);

        x = max(min(x,1),-1);

        Xbkg(:,:,:,ind) = afe.computeSpectrogram(x);

        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end
%Xbkg = log10(Xbkg + epsil);

numTrainBkg = floor(0.70*numBkgClips);
numValTestBkg = floor(0.15*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
YTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValTestBkg) = Xbkg(:,:,:,numTrainBkg+1:numTrainBkg+numValTestBkg);
YValidation(end+1:end+numValTestBkg) = "background";

XTest(:,:,:,end+1:end+numValTestBkg) = Xbkg(:,:,:,numTrainBkg+numValTestBkg+1:numTrainBkg+2*numValTestBkg);
YTest(end+1:end+numValTestBkg) = "background";

%% Display Category repartition

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])

subplot(3,1,1)
histogram(YTrain)
title("Training Label Distribution")

subplot(3,1,2)
histogram(YValidation)
title("Validation Label Distribution")

subplot(3,1,3)
histogram(YTest)
title("Test Label Distribution")

%% Saving the spectrograms
filename = './spectrogram_'+string(afe.options.numberBands)+'bands.mat';
save(filename,"XTrain","YTrain","XValidation","YValidation","XTest","YTest","feOptions");
disp("File saved");

%% Training the network
XTrain = permute(squeeze(XTrain),[2,1,3]);
XValidation = permute(squeeze(XValidation),[2,1,3]);
XTest = permute(squeeze(XTest),[2,1,3]);

% Quantization of the input
nbit = 8;
maxXTrain = max(abs(XTrain(:)));
quantXTrain = quantization(XTrain,nbit-1,maxXTrain);
quantXValidation = quantization(XValidation,nbit-1,maxXTrain);
quantXTest = quantization(XTest,nbit-1,maxXTrain);

%Reshaping the data to work with Matlab layers
[N,S,D] = size(quantXTrain);
[~,~,E] = size(quantXValidation);
[~,~,F] = size(quantXTest);
X = cell([D 1]);
for i = 1:D
    temp = quantXTrain(:,:,i);
    X(i,:) = {temp};
end

XV = cell([E 1]);
for i = 1:E
    temp = quantXValidation(:,:,i);
    XV(i,:) = {temp};
end

XT = cell([F 1]);
for i = 1:F
    temp = quantXTest(:,:,i);
    XT(i,:) = {temp};
end

% Training the network
OutputSize = size(categories(YTrain),1);
epochs = 30;
learning_rate = 3e-3;
layers = [ ...
    sequenceInputLayer(N)
    lstmLayer(64,'OutputMode','last')
    fullyConnectedLayer(OutputSize)
    softmaxLayer
    classificationLayer]

miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',learning_rate, ...
    'MaxEpochs',epochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',true, ...
    'ValidationData',{XV,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',25);

net = trainNetwork(X,YTrain,layers,options);
pred = classify(net,XT);
accuracy = sum(pred == YTest)/numel(YTest);
disp("Test accuracy = "+string(accuracy*100)+"%");

%% Quantization of the network
bitWidthActivation = 9;
bitwidthWeigths = 9;

newQuantXTrain = permute(quantXTrain,[1,3,2]);
newQuantXValidation = permute(quantXValidation,[1,3,2]);
newQuantXTest = permute(quantXTest,[1,3,2]);

leoOptions = leoTrainingOptions('hiddenUnitNumber',length(net.Layers(2).HiddenState),...
                             'miniBatchSize',options.MiniBatchSize,...
                             'maxEpoch',options.MaxEpochs,...
                             'shuffle','every-epoch',...
                             'learningRate',options.InitialLearnRate,...
                             'bitwidth_a',bitWidthActivation-1,...
                             'bitwidth_w',bitwidthWeigths-1);
lstmModel = LeoLstm(leoOptions);
lstmModel.init(net.Layers(2).InputWeights, net.Layers(2).RecurrentWeights, net.Layers(2).Bias, net.Layers(3).Weights, net.Layers(3).Bias);
initAcc = lstmModel.predict(newQuantXTest,YTest);
assert(initAcc == accuracy,"The network you try to quantize is not the same you created");

maxWeight = max([max(net.Layers(2).InputWeights(:)) max(net.Layers(2).RecurrentWeights(:)) max(net.Layers(3).Weights(:))]);
minWeight = min([min(net.Layers(2).InputWeights(:)) min(net.Layers(2).RecurrentWeights(:)) min(net.Layers(3).Weights(:))]);
absMax = max([maxWeight abs(minWeight)]);
disp('Max Weights: '+string(absMax));
absMaxActivation = max([max(abs(lstmModel.states.h(:))) max(abs(lstmModel.states.c(:)))]);
disp('Max Activation: '+string(absMaxActivation));

degradingCoef = 0.01;
maxAccLoss = 0.02;

initMax = absMax;
listAcc = {};
listMax = {};
acc = initAcc;
while acc > (accuracy-maxAccLoss)
    lstmModel.updateBitwidth(absMax);
    acc = lstmModel.predict(newQuantXTest,YTest);
    listAcc{end+1} = acc;
    listMax{end+1} = absMax;
    disp(acc);
    absMax = absMax - (degradingCoef*absMax);
    lstmModel.init(net.Layers(2).InputWeights, net.Layers(2).RecurrentWeights, net.Layers(2).Bias, net.Layers(3).Weights, net.Layers(3).Bias);
    if (acc >= accuracy)
        break;
    end
end
listAcc = cell2mat(listAcc);
listMax = cell2mat(listMax);
[acc,maxIdx] = max(listAcc);
finalMax = listMax(maxIdx);

lstmModel.init(net.Layers(2).InputWeights, net.Layers(2).RecurrentWeights, net.Layers(2).Bias, net.Layers(3).Weights, net.Layers(3).Bias);
lstmModel.updateBitwidth(finalMax);
acc = lstmModel.predict(newQuantXTest,YTest);
tempAcc = acc;
disp("Acc with Quantized Weigths only: "+string(acc*100)+"%");

maxAccLoss = 0.02;
degradingCoef = 0.01;
listAccActivation = {};
listMaxActivation = {};
while acc > (accuracy-maxAccLoss)
    acc = lstmModel.predictQuant(newQuantXTest,YTest,absMaxActivation,maxXTrain);
    disp(acc);
    listAccActivation{end+1} = acc;
    listMaxActivation{end+1} = absMaxActivation;
    absMaxActivation = absMaxActivation - (absMaxActivation*degradingCoef);
    if (acc >= accuracy) | (acc >= tempAcc)
        break;
    end
end

listAccActivation = cell2mat(listAccActivation);
listMaxActivation = cell2mat(listMaxActivation);
[maxAcc,maxIdx] = max(listAccActivation);
finalMaxActivation = listMaxActivation(maxIdx);

acc = lstmModel.predictQuant(newQuantXTest,YTest,finalMaxActivation,maxXTrain);
disp('Final Accuracy: '+string(acc*100)+'%');
disp('Accuracy Loss due to quantization: '+string((initAcc-acc)*100)+'%');
