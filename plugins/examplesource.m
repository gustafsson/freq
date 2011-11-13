function examplesource()
disp('Couldn''t create source. Refer to logfile');
end

function settings=examplesource_settings()
% define this function for all plugins
settings.argumentdescription = 'Signal length [s]';
settings.arguments = '';
end

function data=examplesource_source(p)
if 0 == nargin
  settings.argumentdescription = 'Signal length [s]';
  return
elseif p<0
  data.argumentdescription = 'Signal length [s] (must be positive)';
  return
end

data.samplerate = 44100;
data.data = rand(data.samplerate*p, 1);
end
