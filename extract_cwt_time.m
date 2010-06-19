% extract chunks from a sound file, given start time and end time

function s_cwt = extract_cwt_time(soundfile, starttime, endtime)
    system(['./sonicawe --samples_per_chunk=13 --get_hdf=0 "' soundfile '"' ]);

    data = sawe_loadchunk('save_chunk.h5');
    
    chunk_width = size(data.chunk,1);
    first = floor(starttime*data.samplerate/chunk_width);
    last = ceil(endtime*data.samplerate/chunk_width);

    disp(['Estimated memory needed: ' num2str((last-first+1)*numel(data.chunk)*2*8)/1024/1024 ' MB']);
    
    s_cwt = extract_cwt(soundfile, first, last);

    first_sample = floor(starttime*data.samplerate);
    last_sample = ceil(endtime*data.samplerate);    
    first_sample = first_sample - chunk_size*first;
    last_sample = last_sample - chunk_size*first;
    
    s_cwt = s_cwt(first_sample:last_sample, :);
end
