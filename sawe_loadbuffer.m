function data=sawe_loadbuffer(filename)
% octave
data = load(filename);

% matlab
data.buffer=hdf5read(filename,'buffer');
data.samplerate=hdf5read(filename,'samplerate');
data.offset=hdf5read(filename,'offset');