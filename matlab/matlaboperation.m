function [data,dummy]=matlaboperation(data,dummy)

disp (['doing matlaboperation']);

FS = data.samplerate(1); % data.samplerate is a matrix, not a scalar; hence we need "data.samplerate(1)" instead of "data.samplerate"
offset = data.offset(1);

disp (['FS = ' num2str(FS)]);
disp (['offset = ' num2str(offset)]);
format long
disp (['data size = ' num2str(size(data.buffer))]);
disp (['signal length = ' num2str(numel(data.buffer)/FS)]);

lowpass=0.05;

F=fft(data.buffer);
disp (['fft complete']);
nsave=round(lowpass*numel(F));
F(2+nsave:end-nsave)=0;
disp (['applied filter']);
data.buffer=real(ifft(F));
disp (['ifft complete']);
%endfunction % octave
