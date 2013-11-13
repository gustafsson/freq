function [txt]=sawe_getdatainfo(data)

global state;
if isempty(state)
    state.counter = 1;
else
    state.counter = state.counter + 1;
end

format long

txt = ['#' num2str(state.counter) ' - ' ...
       'data = [' num2str(data.offset/data.fs) ', ' num2str((data.offset+size(data.samples,1))/data.fs) ') s ' ...
       num2str(size(data.samples,1)) ' samples'];

if  1 ~= size(data.samples,2)
    txt = [ txt ' channels=' num2str(size(data.samples,2)) ];
end

if  0 ~= data.overlap
    txt = [ txt ' overlap = ' num2str(data.overlap) ];
end

