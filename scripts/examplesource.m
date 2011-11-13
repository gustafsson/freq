function examplesource_settings()
% define this function for all plugins
end

function data=examplesource_source(p)
if 0 == nargin
  data.data = [];
  return
end

data.samplerate = 44100;
data.data = rand(data.samplerate*p, 1);
end
