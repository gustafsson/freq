function sawe_savechunk(filename, chunk, offset, samplerate)
    save("-hdf5", filename, 'chunk', 'offset', 'samplerate');
endfunction
