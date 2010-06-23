function sawe_savechunk(filename, chunk, offset, samplerate)
% octave
% save('-hdf5', filename, 'chunk', 'offset', 'samplerate');

% matlab
hdf5write(filename,'/chunk',chunk,'/offset',offset,'/samplerate',samplerate);

