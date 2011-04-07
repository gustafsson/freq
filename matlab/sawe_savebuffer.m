function sawe_savebuffer(filename, buffer, offset, samplerate, redundancy, plot)
if exist('OCTAVE_VERSION','builtin')
    %octave
    save('-hdf5', filename, 'buffer', 'offset', 'samplerate', 'redundancy', 'plot');
else
    % matlab
    hdf5write(filename,'/buffer',buffer,'/offset',offset,'/samplerate',samplerate,'/redundancy',redundancy,'/plot',plot);
end

