function data=sawe_loadchunk(filename)
% octave
%data = load(filename);

% matlab
data.chunk=hdf5read(filename,'chunk');
data.samplerate=hdf5read(filename,'samplerate');
data.offset=hdf5read(filename,'offset');