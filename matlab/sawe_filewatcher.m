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
  arguments=[];
end
if nargin<4
  dt=0.025;
end

secondaryfunction = false;
try
% this will fail for secondary function (that doesn't have the same name as the basename of the file)
secondaryfunction = ~exist( [func2str(func) '.m'] );
catch
secondaryfunction = true;
end
if ~secondaryfunction && 1 == nargin(func2str(func)) && 0 ~= numel(arguments)
  disp(['Function ' func2str(func) ' only takes 1 argument, ignoring arguments ''' num2str(arguments) '''']);
end

global sawe_plot_data; %matrix for all lines to be plotted.

resultfile=[datafile '.result.h5'];
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

  if datafile_exists || secondaryfunction
    start_waiting_time = clock;

	if secondaryfunction
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

    if secondaryfunction
      data = [];
      try
        if 0 ~= numel(arguments)
          data = func(arguments);
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
        func(data, arguments);
      end
      data = sawe_discard(data);
    else
      if 1 == nargin(func2str(func))
        data = func(data);
      else
        data = func(data, arguments);
      end
    end

	sawe_savestruct(tempfile, data);
    
    if isoctave
      rename(tempfile,resultfile);   % octave
    else
      movefile(tempfile,resultfile); % matlab
    end

    if logginfo
      disp([ sawe_datestr(now, 'HH:MM:SS.FFF') ' saved results']);
    end
    
    if secondaryfunction
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
