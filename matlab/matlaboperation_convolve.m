function [data]=matlaboperation_convolve(data)

disp('matlaboperation_convolve');

filter=[1 -1 1];
%filter=filter/abs(sum(filter));
data.buffer = conv(data.buffer, filter);
data.buffer = data.buffer + rand(size(data.buffer));

data.redundancy = 5;
data = sawe_discard(data);
