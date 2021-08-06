function [model,acc] = leoTrainNetwork(X,label,options)
    model = LeoLstm(options);
    acc = model.train(X,label);
end

