% extract chunks from a sound file

function s_cwt = sawe_extract_cwt(soundfile,first,last)

	s_cwt = [];

	for k=first:last;
		system(['sonicawe --samples_per_chunk_hint=13 --get_hdf=' num2str(k) ' "' soundfile '"' ]);
        
        data = sawe_loadchunk('save_chunk.h5');
        
		s_cwt_one = flipud((data.chunk)');
		s_cwt = [s_cwt s_cwt_one];
	end
end
