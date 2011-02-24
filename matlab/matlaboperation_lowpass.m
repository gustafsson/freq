function [data,state]=matlaboperation_lowpass(data,state)

%% Update local state
% The variable 'state' is preserved between calls to this operation
if isempty(state) || data.offset==0
    state.counter = 1;
else
    state.counter = state.counter + 1;
end


%% Print information
FS = data.samplerate; 
offset = data.offset;
redundancy = data.redundancy;

format long
disp (['matlaboperation #' num2str(state.counter) ' - ' ...
       'data = [' num2str(offset) ', ' num2str(offset+numel(data.buffer)) ') ' num2str(numel(data.buffer)/FS) ' s, ' ...
       'redundancy = ' num2str(redundancy) ]);

plot(data.buffer(1:40:end));


%% Lowpass filtering
lowpass=0.05; % Keep 5% of the lowest frequencies
F=fft(data.buffer);
nsave=round(lowpass*numel(F)/2);
F(1+nsave:end-nsave)=0;
data.buffer=real(ifft(F));


%% Discard some data
% the rough interpolation below only needs one extra sample at the end
data.redundancy = 0.1*FS;
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
