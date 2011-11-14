function [data]=plotwaveform(data)

disp(['plotwaveform ' sawe_getdatainfo(data)]);

for channel=1:size(data.samples,2)
  x = (1:100:size(data.samples,1));
  t = (data.offset + x) / data.fs;
  hz = 1000 + 1000*data.samples(x, channel);
  data.plot(:,:,channel) = [t' hz];
end

