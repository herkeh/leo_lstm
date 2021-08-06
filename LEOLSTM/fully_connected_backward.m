function [dW,dh,db] = fully_connected_backward(dY,h,W)
dW = dY*h';
dh = W'*dY;
db = sum(dY,2);
end