function [data]=convolve(data)

disp(['convolve ' sawe_getdatainfo(data)]);

filter=[1 -1 1];
%filter=filter/abs(sum(filter));
data.samples = conv(data.samples, filter);
data.samples = data.samples + rand(size(data.samples));

data.overlap = 5;
data = sawe_discard(data);

