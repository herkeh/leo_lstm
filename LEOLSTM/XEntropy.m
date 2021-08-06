classdef XEntropy
    %   Detailed explanation goes here
    
    methods        
        function [l,dl] = loss(~,pred,label)
            % label is an array of 0 except at the right prediction index
            % where its 1.
%             m = size(pred,2);
%             label = logical(label);
%             log_likelihood = -log(boundawayfromzero(pred(label)));
%             l = sum(log_likelihood)/m;
%             dl = (-label./boundawayfromzero(pred))./m;
%             pred(label) = pred(label) - 1;
%             dl = pred;
            l = mean((-1) * sum(label .* log(pred)));
            dl = pred - label;
        end
        
    end
end

function xBound = boundawayfromzero(x)
    precision = class(x);
    xBound = x;
    xBound(xBound < eps(precision)) = eps(precision);
end