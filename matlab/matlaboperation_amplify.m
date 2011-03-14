function [data]=matlaboperation_amplify(data)

global state;
if isempty(state) || data.offset==0
    state.counter = 1;
else
    state.counter = state.counter + 1;
end

disp (['amplify #' num2str(state.counter) ' - ' ...
       'data = [' num2str(data.offset/data.samplerate) ', ' num2str((data.offset+numel(data.buffer))/data.samplerate) ') s ' ...
       '[' num2str(data.offset) ', ' num2str(data.offset+numel(data.buffer)) ') ' ...
       'redundancy = ' num2str(data.redundancy) ]);

data.buffer = data.buffer*4;

% plot hz, evenly distributed
data.plot = [2000
             5000
             6000]';

% plot time and hz
data.plot = [1 2000
             2 5000
             4 6000]';

% plot time, hz and amplitude
data.plot = [1 2000 1
             2 5000 2
             4 6000 1]';

