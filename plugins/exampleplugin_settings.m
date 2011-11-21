function settings=exampleplugin_settings()
% When Sonic AWE finds two files with the same primary name but one with the
% postfix _settings it will interpret that file as a plugin.
% Plugins can be accessible through menu options or toolbar buttons in 
% Sonic AWE.
%
% With Octave, this functions 'exampleplugin_settings' can be a subfunction 
% in the same file as the main function for the plugin 'exampleplugin'.
%
% This are the parameters that can be set by a '_settings' function:
%
% settings.chunk_size
%   -1                      Entire signal in one pass.
%   0 (default)             Let Sonic AWE choose depedning on what's needed 
%                           for the moment.
%   (any other number > 0)  Specific number of samples per pass.
%
%
% settings.compute_chunks_in_order
%   false (default)         Let Sonic AWE choose depedning on what's needed 
%                           for the moment.
%   true                    Compute chunks in consecutive order from 
%                           beginning to end.
%
%
% settings.arguments
%   '' (default)            Any text string that is passed as argument to the
%                           script. 
%                           The value will be used equivalent to: 
%                           eval(['scriptname(data,' settings.arguments ');']);
%
%
% settings.argument_description
%   'Arguments' (default)   This text will show up if the user chooses to 
%                           enter arguments to the script.
%
%
% settings.overlap
%   (any number >= 0)       Number of overlapping samples per chunk. This 
%                           number of samples is included on both sides of each
%                           chunk.
%                           Tip: Use data=sawe_discard(data) to discard 
%                           overlapping samples before returning to Sonic AWE.
%
%
% settings.icon
%   '' (default)            When 'settings.icon' is empty or not set this
%                           script will show up in the menu but not as a
%                           button.
%   (source to image file)  When 'settings.icon' is set this script will show
%                           up as a button with this icon in a toolbar in 
%                           Sonic AWE.
%   (other text)            If no image was found this value is used as text
%                           on a button in a toolbar in Sonic AWE.
%
settings.chunk_size = 0;
settings.compute_chunks_in_order = false;
settings.arguments = '4 ';
settings.argument_description = 'Amplitude factor';
settings.overlap = 0;
settings.icon = 'examplepluginicon.png';
end
