function data=sawe_loadbuffer(filename)
if exist('OCTAVE_VERSION','builtin')
    % octave
    data = load(filename);
else
    % matlab
    data.buffer=hdf5read(filename,'buffer');
    data.samplerate=hdf5read(filename,'samplerate');
    data.offset=hdf5read(filename,'offset');
    data.redundancy=hdf5read(filename,'redundancy');
end

