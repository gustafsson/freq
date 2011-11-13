function data=sawe_loadchunk(filename)
if exist('OCTAVE_VERSION','builtin')
  % octave
  data = load(filename);
else
  % matlab, todo use generic h5info (and h5read) instead
  data.chunk=hdf5read(filename,'chunk');
  data.fs=hdf5read(filename,'fs');
  data.offset=hdf5read(filename,'offset');
end

