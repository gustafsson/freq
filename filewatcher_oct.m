% typical usage
%   filewatcher('datafile.hd5', @work)
% with work defined as
%   function work(datafile)
% 
% work is then called with work('datafile.hd5') whenever datafile.hd5 is modified.
%
% filewatcher does not return. Whatever value returned from func is discarded.
%
% Note: one call to stat takes roughly 0.00004s on johan-laptop. So it shouldn't be an issue to invoke stat 20 times per second (dt=0.05).
function C=filewatcher_oct(datafile, func, arguments, dt)

if nargin<2
  error "syntax: filewatcher(datafile, function, arguments, dt). 'arguments' defaults to [], 'dt' defaults to 0.05"
end
if nargin<3
  arguments=[];
end
if nargin<4
  dt=0.05;
end

resultfile=[datafile '.result.h5'];
tempfile=datafile;

disp (['Monitoring ' datafile]);
while 1
  if ~isempty(stat(datafile)) % fast octave version

    disp (['Processing ' datafile]);
	
    %octave
    data=load(datafile); 
    
    [data, arguments]=func(data, arguments);

    % could perhaps use fieldnames(data) somehow to export this data
    if isfield(data,'buffer')
      sawe_savebuffer_oct(tempfile, data.buffer, data.offset, data.samplerate );
    elseif isfield(data,'chunk')
      sawe_savechunk_oct(tempfile, data.chunk, data.offset, data.samplerate );
    end
    
    rename(tempfile,resultfile);   % octave
    
    disp (['Monitoring ' datafile]);
  else
    sleep(dt); % octave
  end
end

endfunction
