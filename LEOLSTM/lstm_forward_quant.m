function [Y,C,G] = lstm_forward_quant(X,W,R,b,c0,y0,n,maxVal,maxValInput)
    % N : Mini Batch Size
    % S : Sequence Length
    % H : Hidden Units
    
    % G : Gate result
    % C : Cell state

    [~,N,S] = size(X);
    H = size(R,2);
    
    G = zeros(4*H,N,S,'like',X);
    C = zeros(H,N,S,'like',X);
    
    [o_idx,f_idx,i_idx,z_idx] = gate_index(H);
    ifo_idx = [i_idx f_idx o_idx];
    
    Y = y0;
    for ts = 1:1:S
        G(:,:,ts) = W*X(:,:,ts) + R*Y + b;
        G(z_idx,:,ts) = tanh(G(z_idx,:,ts));
        G(ifo_idx,:,ts) = sigmoid(G(ifo_idx,:,ts));
%         G(i_idx,:,ts) = sigmoid(G(i_idx,:,ts));
%         G(f_idx,:,ts) = sigmoid(G(f_idx,:,ts));
%         G(o_idx,:,ts) = sigmoid(G(o_idx,:,ts));
        G(:,:,ts) = quantization(G(:,:,ts),n,maxVal);
        if ts == 1
            C(:,:,ts) = G(z_idx,:,ts).*G(i_idx,:,ts) + G(f_idx,:,ts).*c0;
        else
            C(:,:,ts) = G(z_idx,:,ts).*G(i_idx,:,ts) + G(f_idx,:,ts).*C(:,:,ts-1);
        end
        C(:,:,ts) = quantization(C(:,:,ts),n,maxVal);
        Y = tanh(C(:,:,ts)).*G(o_idx,:,ts);
        Y = quantization(Y,n,maxValInput);
    end
end