function T = extract_cwt(filename, t_start, t_stop)

%call sonicawe and get information about number the chunks corresponding to the time
%selection via system(['sonicawe' ARGUMENT]);

Fs;
firstChunk;
lastChunk;
chunkLength;
numFreq;

T1 = zeros(chunkLength,numFreq,2);

for chunk=firstChunk:lastChunk
	system(['sonicawe' '--extraxt_chunk=' num2str(chunk)]);
	T1((chunkLength*chunk+1):(chunkLength*(chunk+1)),:,:) = load('sawe_chunk.h5');
end

T = T1(:,:,1) + i*T1(:,:,2);