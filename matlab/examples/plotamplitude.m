function [data]=plotamplitude(data)

disp(['plotamplitude ' sawe_getdatainfo(data)]);

global amplitude
if isempty(amplitude)
  amplitude=0;
end

mono = sum(data.buffer, 2);
s = 3/data.samplerate;
for k=1:numel(mono)
  % not eactly amplitude, but related to amplitude and simple
  amplitude = (1-s)*amplitude + s*abs(mono(k));
  mono(k) = amplitude;
end

x = 1:100:numel(mono);
t = (data.offset + x') / data.samplerate;
hz = 1000 + 1000*mono(x);
hz2 = 1000 + 1000*ones(size(hz))*mean(mono);
data.plot(:,:,1) = [t hz];
data.plot(:,:,2) = [t hz2];
mean(mono)

