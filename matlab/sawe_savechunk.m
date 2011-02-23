function sawe_savechunk(filename, chunk, offset, samplerate)
if exist('OCTAVE_VERSION','builtin')
  % octave
  save('-hdf5', filename, 'chunk', 'offset', 'samplerate');
else
  % matlab
  hdf5write(filename,'/chunk',chunk,'/offset',offset,'/samplerate',samplerate);
end

