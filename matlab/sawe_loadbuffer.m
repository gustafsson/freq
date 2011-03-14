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

% 'Supposed to be scalars' are exported from Sonic AWE as 1x1 matrice, not scalars.
% Hence we need to take the value by "data.samplerate(1)" instead of "data.samplerate".
data.samplerate=data.samplerate(1);
data.offset=data.offset(1);
data.redundancy=data.redundancy(1);

data.plot=[]

