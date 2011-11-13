function data=exampleplugin(data, p)

disp(['exampleplugin ' sawe_getdatainfo(data)]);

data.buffer = data.buffer * p;
end

function settings=exampleplugin_settings()
%settings.chunk_size = -1; % entire signal in one pass
%settings.chunk_size = 2^14; % specific number of samples per pass
settings.chunk_size = 0; % arbitrary, Sonic AWE can choose depedning on what's needed for the moment. The default i 0.
settings.compute_chunks_in_order = 0; % this script doesn't require chunks to be passed in consecutive order. The default i false.
settings.arguments = '4'; % k=4 as default argument.
settings.overlapping = 0; % this script doesn't require any overlap between chunks. If a script does, it's up to the script do discard any redundant data before returning data to Sonic AWE. The default i 0.
settings.icon = 'examplepluginicon.png';
end

