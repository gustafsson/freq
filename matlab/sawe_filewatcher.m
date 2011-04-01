% typical usage
%   filewatcher('datafile.hd5', @work)
% with work defined as
%   function work(datafile)
% 
% work is then called with work('datafile.hd5') whenever datafile.hd5 is modified.
%
% filewatcher does not return. Whatever value returned from func is discarded.
%
% Note: one call to stat takes roughly 0.00004s on johan-laptop. So it shouldn't be an issue to invoke stat 40 times per second (dt=0.025).
function C=sawe_filewatcher(datafile, func, arguments, dt)

if nargin<2
  error('syntax: filewatcher(datafile, function, arguments, dt). ''arguments'' defaults to [], ''dt'' defaults to 0.05')
end
if nargin<3
  arguments=[];
end
if nargin<4
  dt=0.025;
end

if 1 == nargin(func2str(func)) && 0 ~= numel(arguments)
    disp(['Function ' func2str(func) ' only takes 1 argument, ignoring arguments ''' num2str(arguments) '''']);
end

global sawe_plot_data; %matrix for all lines to be plotted.

resultfile=[datafile '.result.h5'];
tempfile=datafile;
isoctave=0~=exist('OCTAVE_VERSION','builtin');

disp([ datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF') ' script waiting Sonic AWE running script ''' func2str(func) ''' (datafile ' datafile ')']);
tic
while 1


  if isoctave
    datafile_exists = ~isempty(stat(datafile)); % fast octave version
  else
    datafile_exists = exist(datafile,'file'); % matlab and octave
  end

  if datafile_exists
    disp([ datestr(now, 'HH:MM:SS.FFF') ' Processing ''' datafile '''']);

    try	
      if ~isoctave
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
	catch me
	  disp(me)
	  continue
    end
	
	sawe_plot_data = [];

    % 'Supposed to be scalars' are exported from Sonic AWE as 1x1 matrice, not scalars.
    % Hence we need to take the value by "data.samplerate(1)" instead of "data.samplerate".
    % This reduces 1x1 matrices to scalars
    n = fieldnames(data);
    for k=1:numel(n)
        if numel(data.(n{k})) == 1
            v = data.(n{k});
            data.(n{k}) = v(1);
        end
    end

    disp([ datestr(now, 'HH:MM:SS.FFF') ' Sonic AWE running script ''' func2str(func) '''']);
    if 1 == nargin(func2str(func))
        data = func(data);
    else
        data = func(data, arguments);
    end

    % could perhaps use fieldnames(data) somehow to export this data
    if isfield(data,'buffer')
      sawe_savebuffer(tempfile, data.buffer, data.offset, data.samplerate, data.redundancy, sawe_plot_data );
    elseif isfield(data,'chunk')
      sawe_savechunk(tempfile, data.chunk, data.offset, data.samplerate, data.redundancy, sawe_plot_data );
    end
    
    if isoctave
      rename(tempfile,resultfile);   % octave
    else
      movefile(tempfile,resultfile); % matlab
    end
    disp([ datestr(now, 'HH:MM:SS.FFF') ' saved results']);
    
  else
    if isoctave
      sleep(dt); % octave
    else
      pause(dt); % matlab
    end
  end
end

%endfunction
