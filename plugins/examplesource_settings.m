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
%   'Arguments' (default)    This text will show up if the user chooses to enter arguments to the script.
%
%

  settings.argument_description = 'Signal length [s]';
  settings.arguments = '';
  settings.icon = 'Random input';
  settings.is_source = true;
end
