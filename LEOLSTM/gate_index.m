function [output_index, forget_index, input_index, main_index] = gate_index(hiddenUnit)
    input_index = 1:hiddenUnit;
    forget_index = hiddenUnit + 1:2*hiddenUnit;
    main_index = hiddenUnit*2 + 1:3*hiddenUnit;
    output_index = hiddenUnit*3 +1:4*hiddenUnit;
end