function sawe_savebuffer(filename, buffer, offset, samplerate)
    % save('-hdf5', filename, 'buffer', 'offset', 'samplerate'); %octave
    
    % matlab
    hdf5write(filename,'/buffer',buffer,'/offset',offset,'/samplerate',samplerate);
%endfunction %octave
