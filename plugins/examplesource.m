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
