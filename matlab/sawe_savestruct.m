% Saves a variable of type struct in an 'hdf5' file
%   sawe_savestruct('datafile.h5', datastruct)
function sawe_savestruct(filename, data)
if exist('OCTAVE_VERSION','builtin')
    %octave
	N=fieldnames(data);
	for n=1:numel(N)
		eval([N{n} '= data.(N{n});']);
	end
    save('-hdf5', filename, N{:});
else
    % matlab
	N=fieldnames(data);
	D={}
	for n=1:numel(N); D(1:2,n) = {['/' N{n}]; data.(N{n})}; end
    hdf5write(filename,D{:});
end

