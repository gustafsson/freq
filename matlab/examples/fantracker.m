function [data]=fantracker(data, windowsize)

global state;
if isempty(state) || data.offset==0
    state.counter = 1;
else
    state.counter = state.counter + 1;
end

if nargin==1 || isempty(windowsize)
    windowsize = 1024;
end

disp (['fantracker #' num2str(state.counter) ' - ' ...
       'data = [' num2str(data.offset/data.samplerate) ', ' num2str((data.offset+numel(data.buffer))/data.samplerate) ') s ' ...
       '[' num2str(data.offset) ', ' num2str(data.offset+numel(data.buffer)) ') ' ...
       'redundancy = ' num2str(data.redundancy) ]);

C = stft(data.buffer, windowsize, windowsize, windowsize/2, 3);
C = log(1 + abs(C));
C = stft(data.buffer, windowsize, windowsize, windowsize/2, 3);

max_hz = 1000;
min_hz = 50;
clip_index = round(data.samplerate/max_hz);
C(1:clip_index, :) = 0;
clip_index = round(data.samplerate/min_hz);
clip_index = min(clip_index, windowsize/2);
C(clip_index:end, :) = 0;

[a,j]=max(abs(C));

hz = data.samplerate*(1./j);
t = linspace(data.offset + windowsize/2, data.offset + numel(data.buffer) - windowsize/2, numel(hz))*(1./data.samplerate);
a = min(1, a * (200/windowsize));

data.plot = [t;hz;a];

