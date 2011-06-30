% computes CWT of a signal
function s_cwt = sawe_compute_cwt(X)

	s_cwt = [];
	soundfile = tempname();
	wavwrite(X, 44100, soundfile);

	lastChunk = floor(2^(log2(size(X,1))-13));

	for k=0:lastChunk
		system(['sonicawe --min_hz=30 --wavelet_scale_support=1000 --samples_per_chunk_hint=13 --get_hdf=' num2str(k) ' "' soundfile '"' ]);
		outputfile = ['sonicawe-' num2str(k) '.h5'];
		if exist(outputfile, 'file')
			data = sawe_loadchunk(outputfile);
			delete(outputfile);
			s_cwt = [s_cwt; data.chunk];
		else
			disp(['No results from chunk k=' num2str(k) ', lastChunk=' num2str(lastChunk)]);
		end
	end

	if size(s_cwt,1) > size(X,1)
		s_cwt = s_cwt(1:size(X,1),:);
	end
    delete(soundfile);
end

