function data=exportpeakfrequencies(data, N, filename)
  disp(['exportpeakfrequencies ' sawe_getdatainfo(data)]);

  
  if ~exist('OCTAVE_VERSION', 'builtin')
    error('exportpeakfrequencies does only support Octave so far');
  end


  if nargin<3
    error('syntax: exportpeakfrequencies( N, filename ). Missing number of frequencies and filename to export to.')
  end


  % Preallocate result matrix
  maxAmplitudes = zeros(N,size(data.samples,2));

  for k = 1:size(data.samples,2)
    % compute a spectrogram with a hamming window, 4 times overlapping
    overlap = 4;
    [Y, C] = stft(data.samples(:,k),(N-1)*2,round((N-1)*2/overlap),N-1,'hamming');

    % select frequencies that we are interrested in
    Y = Y(1:N,:)./sqrt(size(Y,1));

    % find max values for each bin
    maxAmplitudes(:,k) = max(abs(Y),[],2);
  end


  % Compute which frequencies we're looking at, includes 0 hz and nyquist freq
  Hz = linspace(0, data.fs/2, N)';


  % Save results
  HzMax = [Hz maxAmplitudes];
  if exist('OCTAVE_VERSION', 'builtin')
    save('-text', filename, 'HzMax');
  else
    error('exportpeakfrequencies does only support Octave so far');
  end

  disp(['Saved matrix of size(' num2str(size(HzMax)) ') to ' filename '.']);


  % Plot results in octave
  plot(Hz, maxAmplitudes);

  
  % Plot results in Sonic AWE
  t0 = linspace(0,2/data.fs,N); % points are unique on the time axis for each line

  for k = 1:size(data.samples,2)
    t = t0 + 0;
    % t = t0 + k; % separates them, might be easier for comparing results
    sawe_plot2(t, Hz, maxAmplitudes(:,k));
  end
end


function settings=exportpeakfrequencies_settings()
  settings.chunk_size = -1; % entire signal in one pass
  settings.arguments = '129, ''peaks.csv''';
  settings.argument_description = 'Number of frequencies and filename. Example: "129, ''peaks.csv''"';
  settings.icon = 'Export frequencies';
end

