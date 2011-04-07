function [data]=plotalarm(data, limit)

disp(['plotalarm ' sawe_getdatainfo(data)]);

global amplitude
if isempty(amplitude)
  amplitude=0;
end

if nargin<2 || isempty(limit)
  limit = 1.78385745242489e-04;
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
hz = 1000*ones(size(x'));
a = mono(x)<limit;
data.plot = [t hz a];

