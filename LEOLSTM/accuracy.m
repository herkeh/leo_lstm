function acc = accuracy(pred,label)
acc = sum(pred == label)/numel(pred);
end

