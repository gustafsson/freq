function [data, d]=matlaboperation_amplify(data, d)

data.buffer = data.buffer*2;

% no redundant data needed by this filter, so we don't need to discard data even if the user selected some redundancy
% data = sawe_discard(data);

%endfunction % octave
