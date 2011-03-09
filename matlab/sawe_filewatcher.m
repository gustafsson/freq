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
function C=sawe_filewatcher(datafile, func, arguments, dt)

if nargin<2
  error("syntax: filewatcher(datafile, function, arguments, dt). 'arguments' defaults to [], 'dt' defaults to 0.05")
end
if nargin<3
  arguments=[];
end
if nargin<4
  dt=0.05;
end

resultfile=[datafile '.result.h5'];
tempfile=datafile;
isoctave=0~=exist('OCTAVE_VERSION','builtin');
%disp (['Monitoring ' datafile]);
while 1
  if isoctave
    datafile_exists = ~isempty(stat(datafile)); % fast octave version
  else
    datafile_exists = exist(datafile,'file'); % matlab and octave
  end

  if datafile_exists
    %disp (['Processing ' datafile]);
	
    if ~isoctave
      pause(0.1); % matlab, wait for slow file system in windows to finish the move

      info=hdf5info(datafile);
      [dset1]=info.GroupHierarchy.Datasets.Name;
      if strcmp(dset1,'/buffer')
          data = sawe_loadbuffer(datafile);
      else
          data = sawe_loadchunk(datafile);
      end
    else
      %octave
      data = load(datafile); 
    end
    
    data = func(data, arguments);

    % could perhaps use fieldnames(data) somehow to export this data
    if isfield(data,'buffer')
      sawe_savebuffer(tempfile, data.buffer, data.offset, data.samplerate, data.redundancy );
    elseif isfield(data,'chunk')
      sawe_savechunk(tempfile, data.chunk, data.offset, data.samplerate, data.redundancy );
    end
    
    if isoctave
      rename(tempfile,resultfile);   % octave
    else
      movefile(tempfile,resultfile); % matlab
    end
    
    %disp (['Monitoring ' datafile]);
  else
    if isoctave
      sleep(dt); % octave
    else
      pause(dt); % matlab
    end
  end
end

endfunction
