function data=sawe_loadstruct(filename)
if exist('OCTAVE_VERSION','builtin')
    % octave
    data = load(filename);
else
    info=hdf5info(filename);
    for k=1:numel(info.GroupHierarchy.Datasets)
        name = info.GroupHierarchy.Datasets(k).Name(2:end); % strip leading '/'
        val=hdf5read(filename, name);
        if isa(val, 'hdf5.h5string')
            val = val.Data;
        end
        data.(name) = val;
    end
end

