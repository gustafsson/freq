function [data,state]=matlabfilter(data, state)
% State can be used for anything you like, you can make it a scalar, matrix, cell, struct, function pointer or whatever.
% Here we use it to count the number of times 'matlabfilter' has been called within this instance.
% Note that state is not global, it is stored per instance. So if 'matlabfilter' is applied two times in Sonic AWE
% two instances will be created.
if numel(state)==0
  state=0;
endif
state = state+1;

% Get transform chunk
T = data.chunk;
FS = data.samplerate(1); % data.samplerate is a matrix, not a scalar

% Width and height of transform chunk
[w h] = size(T);

%disp (['data size = ' num2str(size(T))]);
%disp (['signal length = ' num2str(w/FS)]);

% Apply thresholding, set everything to zero that has an amplitude less than 2 (the transform consists of complex values).
T(abs(T)<0.01)=0;

% Define square in which we will set all transform values to zero
startTime = 1.5;
endTime = 2.5;
startRowf = 2/5; % Fraction of entire height
endRowf = 1/2;

% Compute what that sqaure means in element numbers
startSample = startTime*FS;
endSample = endTime*FS;
startRow = round(h*startRowf);
endRow = round(h*endRowf);

% Create vector that describes which sample number that corresponds to each column in the chunk
ptM=(0:w-1) + data.offset(1);

% Set elements to zero
T((ptM > startSample) & (ptM < endSample), startRow:endRow)=0;

% Return result
data.chunk=T;
endfunction
