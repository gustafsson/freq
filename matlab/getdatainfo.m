function [txt]=getdatainfo(data)

global state;
if isempty(state) || data.offset==0
    state.counter = 1;
else
    state.counter = state.counter + 1;
end

format long

txt = ['#' num2str(state.counter) ' - ' ...
       'data = [' num2str(data.offset/data.samplerate) ', ' num2str((data.offset+size(data.buffer,1))/data.samplerate) ') s ' ...
       '[' num2str(data.offset) ', ' num2str(data.offset+size(data.buffer,1)) ')'];
if  0 ~= data.redundancy
    txt = [ txt ' redundancy = ' num2str(data.redundancy) ];
end

if  1 ~= size(data.buffer,2)
    txt = [ txt ' channels=' num2str(size(data.buffer,2)) ];
end

