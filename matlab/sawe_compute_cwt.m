% computes CWT of a signal
function s_cwt = sawe_compute_cwt(X)

	s_cwt = [];
	soundfile = tempname();
	wavwrite(X, 44100, soundfile);

	for k=0:floor(log2(size(X,1)/13));
		system(['sonicawe --samples_per_chunk_hint=13 --get_hdf=' num2str(k) ' "' soundfile '"' ]);
        outputfile = ['save_chunk-' num2str(k) '.h5'];
        data = sawe_loadchunk(outputfile);
        delete(outputfile);
		s_cwt_one = flipud((data.chunk)');
		s_cwt = [s_cwt s_cwt_one];
	end

    delete(soundfile);
end

