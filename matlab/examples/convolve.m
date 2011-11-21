function [data]=convolve(data)

disp(['convolve ' sawe_getdatainfo(data)]);

filter=[1 -1 1]';
%filter=filter/abs(sum(filter));
for channel=1:size(data.samples,2)
  convdata(:,channel) = conv(data.samples(:,channel), filter);
end

data.samples = convdata;
% play around
data.samples = data.samples + rand(size(data.samples));

data.overlap = 5;
data = sawe_discard(data);

