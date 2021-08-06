classdef Spec
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        miniBatchSize
        hiddenUnitNumber
        learningRate
        validationData
        validationFrequency
        verbose
        shuffle
        maxEpoch
        bitwidth_w
        bitwidth_a
        
    end
    
    methods
        function this = Spec(inputArguments)
            this.miniBatchSize = inputArguments.miniBatchSize;
            this.hiddenUnitNumber = inputArguments.hiddenUnitNumber;
            this.learningRate = inputArguments.learningRate;
            this.validationData = inputArguments.validationData;
            this.validationFrequency = inputArguments.validationFrequency;
            this.verbose = inputArguments.verbose;
            this.shuffle = inputArguments.shuffle;
            this.maxEpoch = inputArguments.maxEpoch; 
            this.bitwidth_w = inputArguments.bitwidth_w;
            this.bitwidth_a = inputArguments.bitwidth_a;
        end
    end
    
    methods (Static)
        function inputArguments = parseInputArguments(varargin)
            p = createParser();
            p.parse(varargin{:})
            inputArguments = convertToArguments(p);
            
        end
    end
end

function p = createParser()
    p = inputParser();
    p.KeepUnmatched = true;
    
    defaultMiniBatchSize = 50;
    defaultHiddenUnitNumber = 100;
    defaultLearningRate = 0.0001;
    defaultValidationData = [];
    defaultValidationFrequency = 50;
    defaultVerbose = true;
    defaultShuffle = 'Once';
    defaultMaxEpoch = 10;
    defaultBitwidth = 32;
    
    p.addParameter('miniBatchSize',defaultMiniBatchSize,@iAssertIsPositiveIntegerScalar);
    p.addParameter('hiddenUnitNumber',defaultHiddenUnitNumber,@iAssertIsPositiveIntegerScalar);
    p.addParameter('learningRate',defaultLearningRate);
    p.addParameter('validationData',defaultValidationData);
    p.addParameter('validationFrequency',defaultValidationFrequency,@iAssertIsPositiveIntegerScalar);
    p.addParameter('verbose',defaultVerbose,@iAssertValidVerbose);
    p.addParameter('shuffle',defaultShuffle,@(x)any(iAssertAndReturnValidShuffleValue(x)));
    p.addParameter('maxEpoch',defaultMaxEpoch,@iAssertIsPositiveIntegerScalar);
    p.addParameter('bitwidth_w',defaultBitwidth,@iAssertIsPositiveIntegerScalar);
    p.addParameter('bitwidth_a',defaultBitwidth,@iAssertIsPositiveIntegerScalar);
end

function inputArguments = convertToArguments(parser)
    results = parser.Results;
    inputArguments = struct;
    inputArguments.miniBatchSize = results.miniBatchSize;
    inputArguments.hiddenUnitNumber = results.hiddenUnitNumber;
    inputArguments.learningRate = results.learningRate;
    inputArguments.validationData = results.validationData;
    inputArguments.validationFrequency = results.validationFrequency;
    inputArguments.verbose = results.verbose;
    inputArguments.shuffle = results.shuffle;
    inputArguments.maxEpoch = results.maxEpoch;
    inputArguments.bitwidth_w = results.bitwidth_w;
    inputArguments.bitwidth_a = results.bitwidth_a;
end

function iAssertIsPositiveIntegerScalar(x)
    validateattributes(x,{'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidVerbose(x)
validateattributes(x,{'logical','numeric'}, ...
    {'scalar','binary'});
end

function shuffleValue = iAssertAndReturnValidShuffleValue(x)
expectedShuffleValues = {'never', 'once', 'every-epoch'};
shuffleValue = validatestring(x, expectedShuffleValues);
end

