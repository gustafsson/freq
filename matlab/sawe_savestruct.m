% Saves a variable of type struct in an 'hdf5' file
%   sawe_savestruct('datafile.h5', datastruct)
function sawe_savestruct(filename, data)
if exist('OCTAVE_VERSION','builtin')
    %octave
    % try to avoid name collissions by prefixing local variables with _
    _filename = filename;
    _data = data;
	_N=fieldnames(_data);
	for _n=1:numel(_N)
		eval([_N{_n} '= _data.(_N{_n});']);
	end
    save('-hdf5', _filename, _N{:});
else
    % matlab
	N=fieldnames(data);
	D={}
	for n=1:numel(N); D(1:2,n) = {['/' N{n}]; data.(N{n})}; end
    hdf5write(filename,D{:});
end

