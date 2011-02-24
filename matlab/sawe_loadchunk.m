function data=sawe_loadchunk(filename)
if exist('OCTAVE_VERSION','builtin')
  % octave
  data = load(filename);
else
  % matlab
  data.chunk=hdf5read(filename,'chunk');
  data.samplerate=hdf5read(filename,'samplerate');
  data.offset=hdf5read(filename,'offset');
end

data.samplerate = data.samplerate(1);
data.offset = data.offset(1);
