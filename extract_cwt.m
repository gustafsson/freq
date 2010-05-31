% extract chunks from a sound file

function s_cwt = extract_cwt(soundfile,first,last)

	s_cwt = [];

	for k=first:last;
		system(['./sonicawe --get_hdf=' num2str(k) ' ' soundfile]);
		data = load('sawe_chunk.h5');
		s_cwt_one = flipud((data.chunk)');
		s_cwt = [s_cwt s_cwt_one];
	end
end
