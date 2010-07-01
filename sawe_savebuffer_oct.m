function sawe_savebuffer_oct(filename, buffer, offset, samplerate)
% octave
save('-hdf5', filename, 'buffer', 'offset', 'samplerate');

% matlab
%hdf5write(filename,'/buffer',buffer,'/offset',offset,'/samplerate',samplerate);

