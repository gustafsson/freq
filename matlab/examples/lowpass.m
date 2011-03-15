function [data]=lowpass(data)

%% Print information
disp(['lowpass ' getdatainfo(data)]);


%% Lowpass filtering
lowpass=0.05; % Keep 5% of the lowest frequencies
F=fft(data.buffer);
nsave=round(lowpass*numel(F)/2);
F(1+nsave:end-nsave)=0;
data.buffer=real(ifft(F));


%% Discard some data
% the rough interpolation below needs a couple of extra samples at each end
data.redundancy = 0.1*data.samplerate;
data = sawe_discard(data, data.redundancy, data.redundancy-1);


% it is ok to return an empty matrix if not enough data was given to start with
if isempty(data.buffer)
    disp('returning empty buffer');
    return
end


%% Interpolate between blocks
% try to do a rough interpolation between blocks to avoid discontinuities
data.buffer=data.buffer-linspace(data.buffer(1), data.buffer(end), numel(data.buffer));


%% Always discard all redundant data before returning
% note: 'data = sawe_discard(data)' has a default behaviour of discarding the same samples that were redundant in the input
data = sawe_discard(data, 0, 1);

%endfunction % octave
