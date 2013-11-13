%
% Sonic AWE communicates with this script through a special data structure 
% with these members:
%
% data.fs
%                            The samplerate of the outgoing data.
%
%
% data.samples
%                            The bulk of the data. Each column is represented 
%                            as one channel. So rand(100,2) creates a stereo 
%                            signal.
%
%
% data.offset
%                            The start of the bulk data.samples.
%
%
% data.overlap
%                            How many samples that are overlapping into the 
%                            next bulk. These should be discarded before
%                            returning. Sonic AWE will then send this number
%                            of extra samples into the next invokation of this
%                            script.
%
% This data structure is used by Sonic AWE both to send data to the script and 
% to read results from the script once it's finished. The script may be called
% multiple times by Sonic AWE if the function exampleplugin_settings permits 
% it.
function data=exampleplugin(data, p)

  disp(['exampleplugin ' sawe_getdatainfo(data)]);

  data.samples = data.samples * p;

end
