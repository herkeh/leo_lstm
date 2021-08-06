function [Y] = my_softmax(X)
Y = exp(X-max(X))./sum(exp(X-max(X)));
end