function result = sigmoid(x)

    result = zeros(size(x,1),size(x,2));
    
    for i=1:size(x,1)
        for j=1:size(x,2)
            result(i,j) = 1 / (1 + exp(-x(i,j)));
        end
    end

end