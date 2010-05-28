function [data,state]=matlabfilter(data, state)
if numel(state)==0
  %state=0;
endif

T = data.chunk;
w = size(T,1);
offset = single(data.offset);
T(abs(T)<2)=0;
ptM=(0:w-1) + offset;
T((ptM > 22050) & (ptM < 44100), round(2*end/5):round(end/2))=0;
data.chunk=T;

%state = updated state
endfunction
