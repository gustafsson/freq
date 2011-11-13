function data=sawe_loadbuffer(filename)
if exist('OCTAVE_VERSION','builtin')
    % octave
    data = load(filename);
else
    % matlab, todo use generic h5info (and h5read) instead
    data.samples=hdf5read(filename,'samples');
    data.fs=hdf5read(filename,'fs');
    data.offset=hdf5read(filename,'offset');
    data.redundancy=hdf5read(filename,'redundancy');
end

