function [data,dummy]=matlaboperation(data)
FS = data.samplerate(1); % data.samplerate is a matrix, not a scalar
offset = data.offset(1);

F=fft(data.buffer);
F(round(end/20):round(19*end/20))=0;
data.buffer=real(ifft(F));
endfunction
