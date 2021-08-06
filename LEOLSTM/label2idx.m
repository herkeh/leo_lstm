function label_array = label2idx(label,keyset)
    number_of_categories = numel(keyset);
    number_of_label = numel(label);
    label_array = zeros(number_of_categories,number_of_label);
    label_index = zeros(1,number_of_label);
    valueSet = 1:1:number_of_categories;
    M = containers.Map(keyset,valueSet);
    for i = 1:1:number_of_label
    label_array(M(string(label(i))),i) = 1;
    [~, max_index] = max(label_array(:,i));
    label_index(i) = max_index;
    end
end

