% Saves a variable of type struct in an 'hdf5' file
%   sawe_savestruct('datafile.h5', datastruct)
function sawe_savestruct(filename, data)
if exist('OCTAVE_VERSION','builtin')
    %octave
    % try to avoid name collissions by prefixing local variables with t_
    t_filename = filename;
    t_data = data;
	t_N=fieldnames(t_data);
	for t_n=1:numel(t_N)
		eval([t_N{t_n} '= t_data.(t_N{t_n});']);
	end
    save('-hdf5', t_filename, t_N{:});
else
    % matlab
	N=fieldnames(data);
	D=cell(2,numel(N));
	for n=1:numel(N)
		val = data.(N{n});
		if islogical(val)
			val = real(val);
		end
		D(1:2,n) = {['/' N{n}]; val};
	end
    hdf5write(filename,D{:});
end

