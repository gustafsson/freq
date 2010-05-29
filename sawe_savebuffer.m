function sawe_savebuffer(filename, buffer, offset, samplerate)
    save("-hdf5", filename, 'buffer', 'offset', 'samplerate');
endfunction
