function newValue = quantization(value,n,maxVal)
%[quantMatrix] = quantizeWeight(n,inputMatrix) 
%   This function quatized the input matrix with the number of bit n.
    valueSign = value<0;
    valueNormalized = abs(value)/maxVal;
    valueNormalized(valueNormalized>1)=1;
    newValue = (round(valueNormalized.*(2^n-1))./(2^n-1)).*maxVal.*(-1).^valueSign;
end

