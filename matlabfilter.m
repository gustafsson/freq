function [data,state]=matlabfilter(data, state)
if numel(state)==0
  %state=0;
endif

T = data.chunk;
FS = data.samplerate(1);
w = size(T,1);
T(abs(T)<2)=0;
ptM=(0:w-1) + data.offset(1);
T((ptM > .5*FS) & (ptM < FS), round(2*end/5):round(end/2))=0;
data.chunk=T;

%state = updated state
endfunction
