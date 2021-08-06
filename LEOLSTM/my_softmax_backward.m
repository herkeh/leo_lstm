function [dY] = my_softmax_backward(Z,dZ)
%     dotProduct = sum(Z.*dZ, 3);
%     dY = dZ - dotProduct;
%     dY = dY.*Z;
    dY = zeros(size(dZ));
    for i = 1:1:size(dZ,1)
        for j = 1:1:size(dZ,1)
            if i == j
                dY(i,:) = dY(i,:) + Z(i,:).*(1-Z(j,:)).*dZ(j,:);
            else
                dY(i,:) = dY(i,:) -(Z(i,:).*Z(j,:)).*dZ(j,:);
            end
        end
    end
%     for i = 1:1:size(dZ,1)
%         for j = 1:1:size(dZ,1)
%             if i == j
%                 dY(i,:) = dY(i,:) + (exp(dZ(i,:))./sum(exp(dZ),1)).*(1 - exp(dZ(j,:))./sum(exp(dZ),1));
%             else
%                 dY(i,:) = dY(i,:) -(exp(dZ(j,:))./sum(exp(dZ),1)).*(exp(dZ(j,:))./sum(exp(dZ),1));
%             end
%         end
%     end
end

