function sawe_savechunk(filename, chunk, offset, samplerate)
    % save('-hdf5', filename, 'chunk', 'offset', 'samplerate'); % octave

    % matlab
    hdf5write(filename,'/buffer',buffer);
    hdf5write(filename,'/offset',offset);
    hdf5write(filename,'/samplerate',samplerate);
%endfunction %octave
