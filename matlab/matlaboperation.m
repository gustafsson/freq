function [data,dummy]=matlaboperation(data,dummy)
FS = data.samplerate(1); % data.samplerate is a matrix, not a scalar; hence we need "data.samplerate(1)" instead of "data.samplerate"
offset = data.offset(1);

lowpass=0.05;

F=fft(data.buffer);
nsave=round(lowpass*numel(F));
F(1+nsave:end-nsave)=0;
data.buffer=real(ifft(F));
%endfunction % octave
