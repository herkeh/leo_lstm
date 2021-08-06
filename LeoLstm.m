classdef LeoLstm < handle
    %LeoLstm 
    %   Cette classe représente la Lstm construite dans le cas de nos
    %   rehcerches. Elle est très simple, mais contrairement au model
    %   Matlab, nous pouvons agir sur des points précis de son
    %   fonctionnement et aussi modifier des aspects comme le nombre de bit
    %   servant à coder ses vecteurs mémoires.
    %   PROPERTIES
    %   options : Objet qui réuni les paramètres nécessaires à
    %   l'entrainement du réseau. Voir la classe Spec.
    %   weights : Structure qui réunie les différents poids de la LSTM.
    %           W - Matrice de poids entre l'entrée et les gates.
    %           R - Matrice de poids entre le hidden_state et les gates
    %           b - Matrice des biais ajoutés avant les gates.
    %           Wfc - Matrice de poids entre la couche softmax et la couche
    %           Fully Connected.
    %           bfc - Matrice des biais ajoutés avant la couche Fully
    %           Connected.
    %   METHODS
    %   train : Cette fonction permet d'entrainer le réseau. Elle appelle
    %   toutes les fonctions intermédiaires de la LSTM.
    %   predict : Cette fonction permet d'utiliser un réseau déja entrainer
    %   pour faire des prédictions.
    
    properties
        options %Classe réunissant les paramètres nécessaires à l'entrainement du réseau.
        weights %Poids de la LSTM
        states  %Hidden State
    end
    
    methods
        function obj = LeoLstm(options)
            % Initialisation des propriétés. Le réseau n'est pas entrainer
            % donc il n'a pas encore de poids. Appelé la fonction train
            % permet d'initialiser les poids en fonction de la taille des
            % entrées.
            obj.options = options;
            obj.weights = [];
        end
    
        function [accEnd,lossEnd] = train(obj,X,label)
            addpath ./LEOLSTM
            
            % Recherche des catégories possible en sortie du réseaux.
            categories_label = categories(label);
            outputNumber = numel(categories_label); % Nombre de classe de sortie possible = Nombre de sortie
            keyset = sort(categories_label(:)); % Le keyset est équivalent à la liste des classes de sorties possibles trié par ordre alphabétique.
            [inputNumber,N,~] = size(X); % Calcul du nombre d'entrée en fonction de la 
            obj.weights = initWeight(inputNumber,outputNumber,obj.options.hiddenUnitNumber);
            miniBatchNumber = ceil(N/obj.options.miniBatchSize);
            obj.states = initState(obj.options.hiddenUnitNumber,X);
            c0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            y0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            if (obj.options.shuffle == "once") || (obj.options.shuffle == "every-epoch")
                [X,label] = shuffle_dataset(X,label);
            end
            step = obj.options.maxEpoch*miniBatchNumber;
            accArray = zeros(1,step);
            lossArray = zeros(1,step);

            plotIndex = 1;
            epochs = 1:1:step;
            figure()
            set(gcf,'Visible','on')
            for e = 1:1:obj.options.maxEpoch
                startIndex = 1;
                stopIndex = mod(N,obj.options.miniBatchSize);
                if stopIndex == 0
                    stopIndex = obj.options.miniBatchSize;
                end
                for i = 1:1:miniBatchNumber
%                     obj.updateBitwidth(1);
                    XBatch = X(:,startIndex:stopIndex,:);
                    [Y_lstm_forward, C, G] = lstm_forward(XBatch,obj.weights.W,obj.weights.R,obj.weights.b,c0,y0); %OK
                    Y_fullyConnected = fully_connected(Y_lstm_forward,obj.weights.Wfc,obj.weights.bfc);%OK
                    Y = my_softmax(Y_fullyConnected);
                    [loss, dLoss] = XEntropy().loss(Y,label2idx(label(startIndex:stopIndex),keyset));%OK
                    pred = idx2label(Y,keyset);%OK
                    acc = accuracy(pred,label(startIndex:stopIndex));%OK
                    accArray(plotIndex) = acc*100;
                    lossArray(plotIndex) = loss;
                    %disp("Loss : "+loss+" - Acc : "+acc);
                    %pred_array(:,start_idx:stop_idx) = pred;
                    %dZ = my_softmax_backward(Y,dLoss);
                    [dWfc,dh,dbfc] = fully_connected_backward(dLoss,Y_lstm_forward,obj.weights.Wfc);
                    [dX,dW,dR,db] = lstm_backward(XBatch,obj.weights.W,obj.weights.R,obj.weights.b,c0,y0,C,G,dh);
                    obj.applyDiff(dW,dR,db,dWfc,dbfc);
                   % obj.updateBitwidth();
                    
                    startIndex = stopIndex + 1;
                    stopIndex = stopIndex + obj.options.miniBatchSize;
                    subplot(2,1,1);
                    plot(epochs(1:plotIndex),accArray(1:plotIndex),'-r');
                    xlim([0,epochs(plotIndex)]);
                    ylim([0,100]);
                    title('Accuracy (%)');
                    
                    subplot(2,1,2);
                    plot(epochs(1:plotIndex),lossArray(1:plotIndex),'-r');
                    ylim([0,-log(1/outputNumber)+0.5]);
                    xlim([0,epochs(plotIndex)]);
                    title('Loss (CrossEntropy)');
                    drawnow;
                    plotIndex = plotIndex + 1;
                end
                if obj.options.shuffle == "every-epoch"
                    [X,label] = shuffle_dataset(X,label);
                end
            end
            obj.states.c = C;
            obj.states.h = Y_lstm_forward;
            accEnd = accArray(end);
            lossEnd = lossArray(end);
            rmpath ./LEOLSTM
        end
        
        function accuracy = predict(obj,X,label)
            addpath ./LEOLSTM;
            obj.states = initState(obj.options.hiddenUnitNumber,X);
            keyset = categories(label);
            outputNumber = numel(keyset);
            c0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            y0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            dataNumber = size(X,2);
            predIndex = zeros(outputNumber,dataNumber);
            miniBatchNumber = ceil(dataNumber/obj.options.miniBatchSize);
            startIndex = 1;
            stopIndex = mod(dataNumber,obj.options.miniBatchSize);

            for l = 1:1:miniBatchNumber
                [Y_lstm_forward,C,~] = lstm_forward(X(:,startIndex:stopIndex,:),obj.weights.W,obj.weights.R,obj.weights.b,c0,y0);
                Y_fullyConnected = fully_connected(Y_lstm_forward,obj.weights.Wfc,obj.weights.bfc);
                Y = my_softmax(Y_fullyConnected);
                predIndex(:,startIndex:stopIndex) = Y;
                startIndex = stopIndex + 1;
                stopIndex = stopIndex + obj.options.miniBatchSize;
                obj.states.c = C;
                obj.states.h = Y_lstm_forward;
            end
            pred = idx2label(predIndex,keyset);
            accuracy = sum(pred == label)/numel(label);
            rmpath ./LEOLSTM;
        end
        
                function accuracy = predictQuant(obj,X,label,maxVal,maxValInput)
            addpath ./LEOLSTM;
            obj.states = initState(obj.options.hiddenUnitNumber,X);
            keyset = categories(label);
            outputNumber = numel(keyset);
            c0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            y0 = zeros(obj.options.hiddenUnitNumber,1,'like',X);
            dataNumber = size(X,2);
            predIndex = zeros(outputNumber,dataNumber);
            miniBatchNumber = ceil(dataNumber/obj.options.miniBatchSize);
            startIndex = 1;
            stopIndex = mod(dataNumber,obj.options.miniBatchSize);

            for l = 1:1:miniBatchNumber
                [Y_lstm_forward,C,~] = lstm_forward_quant(X(:,startIndex:stopIndex,:),obj.weights.W,obj.weights.R,obj.weights.b,c0,y0,obj.options.bitwidth_a,maxVal,maxValInput);
                Y_fullyConnected = fully_connected(Y_lstm_forward,obj.weights.Wfc,obj.weights.bfc);
                Y_fullyConnected = quantization(Y_fullyConnected,obj.options.bitwidth_a,maxVal);
                Y = my_softmax(Y_fullyConnected);
                %Y = quantization(Y,obj.options.bitwidth_a,maxVal);
                predIndex(:,startIndex:stopIndex) = Y;
                startIndex = stopIndex + 1;
                stopIndex = stopIndex + obj.options.miniBatchSize;
                obj.states.c = C;
                obj.states.h = Y_lstm_forward;
            end
            pred = idx2label(predIndex,keyset);
            accuracy = sum(pred == label)/numel(label);
            rmpath ./LEOLSTM;
        end
        
        function init(obj,W,R,b,Wfc,bfc)
                weight = struct;
                weight.W = W;
                weight.R = R;
                weight.b = b;
                weight.Wfc = Wfc;
                weight.bfc = bfc;
            obj.weights = weight;
        end

        function applyDiff(obj,dW,dR,db,dWfc,dbfc)
            obj.weights.W = obj.weights.W - dW.*obj.options.learningRate;
            obj.weights.R = obj.weights.R - dR.*obj.options.learningRate;
            obj.weights.b = obj.weights.b - db.*obj.options.learningRate;
            obj.weights.Wfc = obj.weights.Wfc - dWfc.*obj.options.learningRate;
            obj.weights.bfc = obj.weights.bfc - dbfc.*obj.options.learningRate;
        end
        
        function updateBitwidth(obj,maxVal)
%             obj.weights.W(obj.weights.W>1)=1;
%             obj.weights.R(obj.weights.R>1)=1;
%             obj.weights.b(obj.weights.b>1)=1;
%             obj.weights.Wfc(obj.weights.Wfc>1)=1;
%             obj.weights.bfc(obj.weights.bfc>1)=1;
%             
%             obj.weights.W(obj.weights.W<(-1))=(-1);
%             obj.weights.R(obj.weights.R<(-1))=(-1);
%             obj.weights.b(obj.weights.b<(-1))=(-1);
%             obj.weights.Wfc(obj.weights.Wfc<(-1))=(-1);
%             obj.weights.bfc(obj.weights.bfc<(-1))=(-1);
            
            obj.weights.W = quantization(obj.weights.W,obj.options.bitwidth_w,maxVal);
            obj.weights.R = quantization(obj.weights.R,obj.options.bitwidth_w,maxVal);
            obj.weights.b = quantization(obj.weights.b,obj.options.bitwidth_w,maxVal);
            obj.weights.Wfc = quantization(obj.weights.Wfc,obj.options.bitwidth_w,maxVal);
            obj.weights.bfc = quantization(obj.weights.bfc,obj.options.bitwidth_w,maxVal);
        end
        
        function y = pact_quantization(obj,x)
            sgn_x = x<0;
            y_temp = 0.5.*(abs(x)-abs(abs(x)-obj.weights.alpha)+obj.weights.alpha).*(-1).^sgn_x;
            y = quantization(y_temp,obj.options.bitwidth,obj.weigths.alpha);
        end
        
        function d_alpha = pact_quantization_backward(obj,dY)
            d_alpha = dY;
            d_alpha(d_alpha>obj.alpha)=1;
            d_alpha(d_alpha<(-obj.alpha))=-1;
            d_alpha(d_alpha>(-obj.alpha) & d_alpha<obj.alpha)=0;
            
        end
    end
end

function weight = initWeight(inputNumber,outputNumber,hiddenUnitNumber)
    weight = struct;
    weight.W = -0.1 + 0.2*rand(hiddenUnitNumber*4, inputNumber);
    weight.R = -0.1 + 0.2*rand(hiddenUnitNumber*4, hiddenUnitNumber);
    weight.b = -0.1 + 0.2*rand(hiddenUnitNumber*4, 1);
    weight.Wfc = -0.1 + 0.2*rand(outputNumber, hiddenUnitNumber);
    weight.bfc = -0.1 + 0.2*rand(outputNumber,1);
end

function states = initState(hiddenUnit,X)
    states = struct;
    states.c = zeros(hiddenUnit,1,'like',X);
    states.h = zeros(hiddenUnit,1,'like',X);
end

function newValue = changeBitwidth(value,n)
    maxValue = max(value(:));
    valueNorm = value./maxValue;   
    newValue = (round(valueNorm.*(2^n-1))./(2^n-1)).*maxValue;
end

% function newValue = quantization(value,n,maxVal)
%     valueSign = value<0;
%     valueNormalized = abs(value)/maxVal;
%     valueNormalized(valueNormalized>1)=1;
%     newValue = (round(valueNormalized.*(2^n-1))./(2^n-1)).*maxVal.*(-1).^valueSign;
% end

function [X,T] = shuffle_dataset(input,label)
    [~,N,~] = size(input);
    idx = randperm(N);
    X = input(:,idx,:);
    T = label(idx);
end