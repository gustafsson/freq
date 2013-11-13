function [data]=lowpass(data)

%% Print information
disp(['lowpass ' sawe_getdatainfo(data)]);


%% Lowpass filtering
lowpass=0.05; % Keep 5% of the lowest frequencies
F=fft(data.samples);
nsave=round(lowpass*numel(F)/2);
F(1+nsave:end-nsave)=0;
data.samples=real(ifft(F));


%% Discard some data
% the rough interpolation below needs a couple of extra samples at each end
data.fs = 0.1*data.fs;
data = sawe_discard(data, data.overlap, data.overlap-1);


% it is ok to return an empty matrix if not enough data was given to start with
if isempty(data.samples)
    disp('returning empty buffer');
    return
end


%% Interpolate between blocks
% try to do a rough interpolation between blocks to avoid discontinuities
data.samples=data.samples-linspace(data.samples(1), data.samples(end), numel(data.samples))';


%% Always discard all redundant data before returning
% note: 'data = sawe_discard(data)' has a default behaviour of discarding the same samples that were redundant in the input
data = sawe_discard(data, 0, 1);

%endfunction % octave
