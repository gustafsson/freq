function [data,dummy]=matlaboperation(data,dummy)

FS = data.samplerate(1); % data.samplerate is a matrix, not a scalar; hence we need "data.samplerate(1)" instead of "data.samplerate"
offset = data.offset(1);

format long
disp (['matlaboperation -  FS = ' num2str(FS) ', offset = ' num2str(offset) ...
       ', data size = ' num2str(numel(data.buffer)) ', signal length = ' num2str(numel(data.buffer)/FS)]);

lowpass=0.05; % Keep 5% of the lowest frequencies

F=fft(data.buffer);
nsave=round(lowpass*numel(F)/2);
F(2+nsave:end-nsave)=0;
data.buffer=real(ifft(F));
%endfunction % octave
