function data=examplesource(p)
%
% data.fs
%                            The samplerate of the outgoing data.
%
%
% data.samples
%                            The bulk of the data. Each column is represented as one channel. So rand(100,2) creates a stereo signal.
%
%
% data.argument_description
%                            May also set a new argument description to be shown if data.samples is empty.
%


  % validate input argument 'p' and update argument_description accordingly
  if 0 == nargin
    data.argument_description = 'Signal length [s]';
    return
  elseif p<0
    data.argument_description = 'Signal length [s] (must be positive)';
    return
  end


  data.fs = 44100;
  data.samples = rand(data.fs*p, 1);
end



function settings=examplesource_settings()
% This function, a '_settings' function, must be defined for all plugins to be automatically discovered by Sonic AWE.
%
%
% settings.arguments
%   '' (default)             Any text string that is passed as argument to the script. 
%                            The value will be used equivalent to: eval(['scriptname(data,' settings.arguments ');']);
%
%
% settings.argument_description
%   'Arguments' (default)    This text will show up if the user is chooses to enter arguments to the script.
%
%

  settings.argument_description = 'Signal length [s]';
  settings.arguments = '';
  settings.icon = 'Random input';
  settings.is_source = true;
end



