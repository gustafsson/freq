function sawe_savechunk(filename, chunk, offset, samplerate)
% octave
% save('-hdf5', filename, 'chunk', 'offset', 'samplerate');

% matlab
hdf5write(filename,'/buffer',buffer);
hdf5write(filename,'/offset',offset);
hdf5write(filename,'/samplerate',samplerate);
