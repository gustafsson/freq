function [data]=plotwaveform(data)

disp(['plotwaveform ' sawe_getdatainfo(data)]);

for channel=1:size(data.buffer,2)
  x = (1:100:size(data.buffer,1));
  t = (data.offset + x) / data.samplerate;
  hz = 1000 + 1000*data.buffer(x,channel);
  data.plot(:,:,channel) = [t' hz];
end

size(data.buffer)

