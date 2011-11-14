function data=reversesignal(data)
  data.samples = flipud(data.samples);
end


function settings=reversesignal_settings()
  settings.chunk_size = -1; % entire signal in one pass
  settings.argument_description = ''; % No argument
  settings.icon = 'Reverse signal';
end

