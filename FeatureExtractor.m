classdef FeatureExtractor < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        options
        frequencyList
        num
        den
    end
    
    methods
        function obj = FeatureExtractor(options)
            %UNTITLED4 Construct an instance of this class
            %   Detailed explanation goes here
            obj.options = options;
            if obj.options.scale == "Log"
                obj.frequencyList = logspace(log10(obj.options.fmin),log10(obj.options.fmax),obj.options.numberBands);
            elseif obj.options.scale == "Mel"
                b = hz2mel([obj.options.fmin,obj.options.fmax]);
                melVect = linspace(b(1),b(2),obj.options.numberBands);
                obj.frequencyList = mel2hz(melVect);
            elseif obj.options.scale == "Bark"
                b = hz2bark([fmin,fmax]);
                melVect = linspace(b(1),b(2),obj.options.numberBands);
                obj.frequencyList = mel2hz(melVect);
            end
            
            a = sqrt(1+(1/(4*(obj.options.q^2))));
            b = 1/(2*obj.options.q);
            for k=1:obj.options.numberBands
               assert(obj.frequencyList(k)*(a+b)/(obj.options.fs/2)<1,"The max frequency for quality factor of : "+string(obj.options.q)+" and sampleing frequency of "+...
                   string(obj.options.fs)+" Hz, is "+string(((obj.options.fs/2))/(a+b))+" Hz");
               [obj.num(k,:),obj.den(k,:)]=butter(obj.options.order,[(obj.frequencyList(k)*(a-b)/(obj.options.fs/2)) (obj.frequencyList(k)*(a+b)/(obj.options.fs/2))], 'bandpass');
            end
        end
        
        function spectrogram = extract(obj,ads)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            segmentSamples = round(obj.options.segmentDuration*obj.options.fs);
            frameSamples = round(obj.options.frameLength*obj.options.fs);
            hopSamples = round(obj.options.hopLength*obj.options.fs);
            overlapSamples = frameSamples - hopSamples;
            numFrames = floor(segmentSamples/overlapSamples) -1;
            
            if ~isempty(ver('parallel')) && obj.options.parallel
                pool = gcp;
                numPar = numpartitions(ads,pool);
            else
                numPar = 1;
            end
            
            padding = obj.options.padding;
            numberBands = obj.options.numberBands;
            num_ = obj.num;
            den_ = obj.den;
            parfor ii = 1:numPar
                subds = partition(ads,numPar,ii);
                XSpect = zeros(numFrames,numberBands,1,numel(subds.Files));
                for idx = 1:numel(subds.Files)
                    x = read(subds);
                    if padding
                        x = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
                    end
                    % Informations available here : http://www.sengpielaudio.com/calculator-cutoffFrequencies.htm

                    npoints=length(x);       % number of points in the recording
                    nframes=floor(npoints/overlapSamples) -1;    % number of frames
                    data_out=zeros(numberBands,npoints);
                    average_power=zeros(numberBands,nframes);

                    for k=1:numberBands
                        data_out(k,:)=filter(num_(k,:),den_(k,:),x);

                    % Initialize Variables
                        samp1 = 1; samp2 = frameSamples; %Initialize frame start and end

                        for i = 1:nframes
                            % Get current frame for analysis
                            frame = data_out(k,samp1:samp2);
                            % calculate average_power
                            average_power(k,i) = mean(frame.^2);
                            % Step up to next frame of speech
                            samp1 = samp1 + overlapSamples;
                            samp2 = samp2 + overlapSamples;
                        end
                    end

                    XSpect(:,:,:,idx) = average_power'./max(abs(average_power(:)));
                    
                end
                XSpectC{ii} = XSpect;
            end 
            spectrogram = cat(4,XSpectC{:});

        end
        
        function XSpect = computeSpectrogram(obj,x)
            frameSamples = round(obj.options.frameLength*obj.options.fs);
            hopSamples = round(obj.options.hopLength*obj.options.fs);
            overlapSamples = frameSamples - hopSamples;

            npoints=length(x);       % number of points in the recording
            nframes=floor(npoints/overlapSamples) -1;    % number of frames
            data_out=zeros(obj.options.numberBands,npoints);
            average_power=zeros(obj.options.numberBands,nframes);

            for k=1:obj.options.numberBands
                data_out(k,:)=filter(obj.num(k,:),obj.den(k,:),x);

            % Initialize Variables
                samp1 = 1; samp2 = frameSamples; %Initialize frame start and end

                for i = 1:nframes
                    % Get current frame for analysis
                    frame = data_out(k,samp1:samp2);
                    % calculate average_power
                    average_power(k,i) = mean(frame.^2);
                    % Step up to next frame of speech
                    samp1 = samp1 + overlapSamples;
                    samp2 = samp2 + overlapSamples;
                end
            end

            XSpect = average_power'./max(abs(average_power(:)));
        end
        
        function plotFrequencyResponse(obj)
            figure();
            for k=1:obj.options.numberBands
                [h,w]=freqz(obj.num(k,:),obj.den(k,:));
                semilogx(w/pi*obj.options.fs/2,20*log10(abs(h)))
                %plot(w/pi*Fs/2/1000,20*log10(abs(h)))
                ax = gca;
                ax.YLim = [-60 0];
                xlabel('Frequency (Hz)')
                ylabel('Magnitude (dB)')
                hold on;
            end
        end
    end
end