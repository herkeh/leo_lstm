function options = featureExtractionOptions(varargin)
    inputArguments = FeatureExtractionSpec.parseInputArguments(varargin{:});
    options =  FeatureExtractionSpec(inputArguments);
end