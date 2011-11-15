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

time_out = 10;

if nargin<2
  error('syntax: filewatcher(datafile, function, arguments, dt). ''arguments'' defaults to [], ''dt'' defaults to 0.05')
end
if nargin<3
  arguments=cell();
end
if nargin<4
  dt=0.025;
end

noinputdata = ~isempty(strfind(datafile, '.result.h5'));
if noinputdata
  fargin = numel(arguments);
  try
    fargin = nargin(func2str(func));
  catch
    % never mind, if 'nargin' can't find the function
  end
  if fargin < numel(arguments)
    error(['Function ' func2str(func) ' takes ' num2str(fargin) ' arguments but ' num2str(numel(arguments)) ' arguments was provided']);
  end
else
  if nargin(func2str(func))-1 ~= numel(arguments)
    error(['Function ' func2str(func) ' takes ' num2str(nargin(func2str(func))-1) ' extra arguments but ' num2str(numel(arguments)) ' arguments was provided']);
  end
end

global sawe_plot_data; %matrix for all lines to be plotted.

if noinputdata
  resultfile = datafile;
  datafile = [resultfile '.tmp'];
else
  resultfile = [datafile '.result.h5'];
end
tempfile=datafile;
isoctave=0~=exist('OCTAVE_VERSION','builtin');

disp([ sawe_datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF') ' Sonic AWE running script ''' func2str(func) ''' (datafile ''' datafile ''')']);
disp(['Working dir: ' pwd]);
tic
logginfo=false;
start_waiting_time = clock;


while 1

  if etime(clock,start_waiting_time) > time_out
    exit;
  end
  
  if isoctave
    datafile_exists = ~isempty(stat(datafile)); % fast octave version
  else
    datafile_exists = exist(datafile,'file'); % matlab and octave
  end

  if datafile_exists || noinputdata
    start_waiting_time = clock;

	if noinputdata
      data = struct();
    else
      if logginfo
        disp([ sawe_datestr(now, 'HH:MM:SS.FFF') ' Processing input']);
      end

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
      catch
        disp(lasterr)
        continue
      end
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

    if logginfo
      disp([ sawe_datestr(now, 'HH:MM:SS.FFF') ' Sonic AWE running script ''' func2str(func) '''']);
    end

    if noinputdata
      data = [];
      try
        if 0 ~= numel(arguments)
          data = func(arguments{:});
        else
          data = func();
        end
      catch
        disp(lasterr);
      end
      if ~isstruct(data)
        data = struct();
        data.dummy = [];
      end
    elseif 0 == nargout(func2str(func))
      if 1 == nargin(func2str(func))
        func(data);
      else
        func(data, arguments{:});
      end
      data = sawe_discard(data);
    else
      if 1 == nargin(func2str(func))
        data = func(data);
      else
        data = func(data, arguments{:});
      end
    end

	data.plot = sawe_plot_data;
	sawe_savestruct(tempfile, data);
    
    if isoctave
      rename(tempfile,resultfile);   % octave
    else
      movefile(tempfile,resultfile); % matlab
    end

    if logginfo
      disp([ sawe_datestr(now, 'HH:MM:SS.FFF') ' saved results']);
    end
    
    if noinputdata
      exit;
    end
  else
    if isoctave
      sleep(dt); % octave
    else
      pause(dt); % matlab
    end
  end
end

%endfunction
