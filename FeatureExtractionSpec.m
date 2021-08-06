classdef FeatureExtractionSpec
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        fs
        fmin
        fmax
        numberBands
        q
        order
        frameLength
        hopLength
        scale
        segmentDuration
        padding
        parallel
        
    end
    
    methods
        function this = FeatureExtractionSpec(inputArguments)
            this.fs = inputArguments.fs;
            this.fmin = inputArguments.fmin;
            this.fmax = inputArguments.fmax;
            this.numberBands = inputArguments.numberBands;
            this.q = inputArguments.q;
            this.order = inputArguments.order;
            this.frameLength = inputArguments.frameLength;
            this.hopLength = inputArguments.hopLength;
            this.scale = inputArguments.scale; 
            this.segmentDuration = inputArguments.segmentDuration;
            this.padding = inputArguments.padding;
            this.parallel = inputArguments.parallel;
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
    
    defaultFs = 16e3;
    defaultFmin = 50;
    defaultFmax = 5e3;
    defaultNumberBands = 16;
    defaultQ = 2;
    defaultOrder = 3;
    defaultFrameLength = 25e-3;
    defaultHopLength = 10e-3;
    defaultScale = "Log";
    defaultSegmentDuration = 1;
    defaultPadding = false;
    defaultParallel = false;
    
    p.addParameter('fs',defaultFs,@iAssertIsPositiveIntegerScalar);
    p.addParameter('fmin',defaultFmin);
    p.addParameter('fmax',defaultFmax);
    p.addParameter('numberBands',defaultNumberBands,@iAssertIsPositiveIntegerScalar);
    p.addParameter('q',defaultQ);
    p.addParameter('order',defaultOrder,@iAssertIsPositiveIntegerScalar);
    p.addParameter('frameLength',defaultFrameLength);
    p.addParameter('hopLength',defaultHopLength);
    p.addParameter('scale',defaultScale,@(x)any(iAssertAndReturnValidScaleValue(x)));
    p.addParameter('segmentDuration',defaultSegmentDuration);
    p.addParameter('padding',defaultPadding,@iAssertValidVerbose);
    p.addParameter('parallel',defaultParallel,@iAssertValidVerbose);
end

function inputArguments = convertToArguments(parser)
    results = parser.Results;
    inputArguments = struct;
    inputArguments.fs = results.fs;
    inputArguments.fmin = results.fmin;
    inputArguments.fmax = results.fmax;
    inputArguments.numberBands = results.numberBands;
    inputArguments.q = results.q;
    inputArguments.order = results.order;
    inputArguments.frameLength = results.frameLength;
    inputArguments.hopLength = results.hopLength;
    inputArguments.scale = results.scale;
    inputArguments.segmentDuration = results.segmentDuration;
    inputArguments.parallel = results.parallel;
    inputArguments.padding = results.padding;
end

function iAssertIsPositiveIntegerScalar(x)
    validateattributes(x,{'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidVerbose(x)
validateattributes(x,{'logical','numeric'}, ...
    {'scalar','binary'});
end

function scaleValue = iAssertAndReturnValidScaleValue(x)
expectedScaleValues = {'Log', 'Bark', 'Mel'};
scaleValue = validatestring(x, expectedScaleValues);
end