function [data]=plotamplitude(data)
toc
disp(['plotamplitude ' sawe_getdatainfo(data)]);
tic
global amplitude
if isempty(amplitude)
  amplitude=0;
end

mono = sum(data.buffer, 2);
downsample = 2^13;
mono_offset = ceil(data.offset/downsample)*downsample;
mono=mono( (mono_offset+1:data.offset + end-mod(data.offset + end, downsample))- mono_offset);
mono = reshape(mono, floor(numel(mono)/downsample), downsample);
mono = max(abs(mono), [], 2);
mono_samplerate = data.samplerate/downsample;
s = 3/mono_samplerate;
for k=1:numel(mono)
  % not exactly amplitude, but related to amplitude and simple
  amplitude = (1-s)*amplitude + s*mono(k);
  mono(k) = amplitude;
end

x = 1:numel(mono);
t = data.offset/data.samplerate + x' / mono_samplerate;
hz = 1000 + 10000*mono(x);
hz2 = 1000 + 10000*mean(mono);
toc
disp('plotting');
sawe_plot(t, hz);
%sawe_plot(t, hz2);
disp(['Mean = ' num2str(mean(mono))]);
tic