function [dX,dW,dR,db] = lstm_backward(X,W,R,b,c0,y0,C,G,dZ)
[~, N, S] = size(X);
H = size(R,2);

[o_idx, f_idx, i_idx, z_idx] = gate_index(H);

dG = zeros(4*H, N, 'like', X);
dX = zeros(size(X), 'like', X);
dW = zeros(size(W), 'like', X);
dR = zeros(size(R), 'like', X);
db = zeros(size(b), 'like', b);

if size(y0, 2) == 1
    y0 = repmat(y0, 1, N);
end

dY = dZ; %OK
tanhC = tanh(C(:,:,S)); %OK
dG(o_idx,:) = dY.*tanhC.*G(o_idx,:,S).*(1 - G(o_idx,:,S)); %OK
dC = dY.*G(o_idx,:,S).*( 1 - tanhC.^2); %OK
dG(f_idx,:) = dC.*C(:,:,S-1).*G(f_idx,:,S).*(1-G(f_idx,:,S));%OK
dG(i_idx,:) = dC.*G(z_idx,:,S).*G(i_idx,:,S).*(1-G(i_idx,:,S));%OK
dG(z_idx,:) = dC.*G(i_idx,:,S).*(1-G(z_idx,:,S).^2);%OK
dX(:,:,S) = W(z_idx,:)'*dG(z_idx,:) + W(i_idx,:)'*dG(i_idx,:) + W(f_idx,:)'*dG(f_idx,:) + W(o_idx,:)'*dG(o_idx,:);%OK
dW = dW + dG*X(:,:,S)';%OK
Ymm = tanh(C(:,:,S-1)).*G(o_idx,:,S-1);%OK
dR = dR + dG*Ymm';%OK
db = db + sum(dG,2);%OK
for ts = (S-1):-1:2 %OK
    dY = R(z_idx,:)'*dG(z_idx,:) + R(i_idx,:)'*dG(i_idx,:) + R(f_idx,:)'*dG(f_idx,:) + R(o_idx,:)'*dG(o_idx,:);%OK
    tanhC = tanh(C(:,:,ts));%OK
    dG(o_idx,:) = dY.*tanhC.*G(o_idx,:,ts).*(1 - G(o_idx,:,ts));%OK
    dC = dY.*G(o_idx,:,ts).*(1-tanhC.^2)+ dC.*G(f_idx,:,ts+1); %OK
    dG(f_idx,:) = dC.*C(:,:,ts-1).*G(f_idx,:,ts).*(1-G(f_idx,:,ts));%OK
    dG(i_idx,:) = dC.*G(z_idx,:,ts).*G(i_idx,:,ts).*(1-G(i_idx,:,ts));
    dG(z_idx,:) = dC.*G(i_idx,:,ts).*(1-G(z_idx,:,ts).^2);%OK
    dX(:,:,ts) = W(z_idx,:)'*dG(z_idx,:) + W(i_idx,:)'*dG(i_idx,:) + W(f_idx,:)'*dG(f_idx,:) + W(o_idx,:)'*dG(o_idx,:);%OK
    dW = dW + dG*X(:,:,ts)';%OK
    Ymm = tanh(C(:,:,ts-1)).*G(o_idx,:,ts-1);%OK
    dR = dR + dG*Ymm';%OK
    db = db + sum(dG,2);%OK
end
dY = R(z_idx,:)'*dG(z_idx,:) + R(i_idx,:)'*dG(i_idx,:) + R(f_idx,:)'*dG(f_idx,:) + R(o_idx,:)'*dG(o_idx,:);%OK
tanhC = tanh(C(:,:,1));%OK
dG(o_idx,:) = dY.*tanhC.*G(o_idx,:,1).*(1 - G(o_idx,:,1));%OK
dC = dY.*G(o_idx,:,1).*(1-tanhC.^2)+ dC.*G(f_idx,:,2);%OK
dG(f_idx,:) = dC.*c0.*G(f_idx,:,1).*(1-G(f_idx,:,1));%OK
dG(i_idx,:) = dC.*G(z_idx,:,1).*G(i_idx,:,1).*(1-G(i_idx,:,1));%OK
dG(z_idx,:) = dC.*G(i_idx,:,1).*(1-G(z_idx,:,1).^2);%OK
dX(:,:,1) = W(z_idx,:)'*dG(z_idx,:) + W(i_idx,:)'*dG(i_idx,:) + W(f_idx,:)'*dG(f_idx,:) + W(o_idx,:)'*dG(o_idx,:);%OK
dW = dW + dG*X(:,:,1)';%OK
Ymm = y0;%OK
dR = dR + dG*Ymm';%OK
db = db + sum(dG,2);%OK
end

