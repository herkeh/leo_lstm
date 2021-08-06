classdef ToyLossLayer
    %TOYLOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = ToyLossLayer()
             %   Detailed explanation goes here
        end
        
        function [l,dl] = loss(obj,pred,label)
            l = sum((pred-label).^2,1);
            l = l ./ (2*size(label,1));
            dl = 2*(pred-label);
        end
    end
end

