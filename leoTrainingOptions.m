function options = leoTrainingOptions(varargin)
    inputArguments = Spec.parseInputArguments(varargin{:});
    options =  Spec(inputArguments);
end

