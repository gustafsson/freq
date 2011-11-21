%
% sawe_plot2, plots lines with varying amplitude in Sonic AWE
%
%   sawe_plot2(t, hz, a)
%
% 't' is a vector with coordinates along the time axis.
% 'hz' is either a vector of the same length as 't' or a scalar.
% 'a' is either a vector of the same length as 't' or a scalar.
%
%
% To plot multiple lines with different coordinates along the time axis, call sawe_plot multiple times or group parameters in pairs:
%
%   sawe_plot(t1, hz1, a1, t2, hz2, a2, ... tn, hzn, an)
%
%
% See also sawe_plot which doesn't require the amplitude argument.
%
function sawe_plot2(varargin);

global sawe_hold_plot;
global sawe_plot_data;

if 0~=mod(nargin,3)
	error('Number of inputs must be a multiple of 3.');
end

max_length = size(sawe_plot_data,1);

for n=1:nargin/3

		t = varargin{3*(n-1) + 1};
		f = varargin{3*(n-1) + 2};
		a = varargin{3*(n-1) + 3};

    length_ = numel(t);
    t = reshape(t,length_,1);

    if( numel(t) == numel(f) && numel(t) == numel(a) );

		  f = reshape(f,length_,1);
		  a = reshape(a,length_,1);

    elseif( numel(t) == numel(f) && numel(a) == 1);

		  f = reshape(f,length_,1);
		  a = a*ones(length_,1);

    elseif( numel(t) == numel(a) && numel(f) == 1);

		  a = reshape(a,length_,1);
		  f = f*ones(length_,1);

    elseif( numel(a) == 1 && numel(f) == 1)

		  a = a*ones(length_,1);
		  f = f*ones(length_,1);

    else
      error('Frequency and amplitude vectors need to be of the same length as time vector, or set as scalars. n: %u --- t: %u, f: %u, a: %u', n, numel(t), numel(f), numel(a));
    end %if

    if isempty(t)
      % grow matrix
      sawe_plot_data(1,3,end+1) = 0;
      continue
    end

	% workaround to cope with lines of different length: repeat the last point to fill the vector!
	% this will not cause problems, as sawe will prune those points anyway.

    if max_length < length_ && max_length>0

      % grow matrix
      sawe_plot_data(length_,3,1) = 0;
      for s=1:size(sawe_plot_data,3)
        sawe_plot_data(max_length+1:length_,:,s) = ones(length_-max_length,1) * sawe_plot_data(max_length,:,s);
      end

    end %if

    max_length = max(max_length, length_);
    if (length_ < max_length)
      t = [t; t(end)*ones( max_length - length_, 1)];
      f = [f; f(end)*ones( max_length - length_, 1)];
      a = [a; a(end)*ones( max_length - length_, 1)];

    end %if

    if 0~=numel(sawe_plot_data)
      zdim = size(sawe_plot_data,3) + 1;
    else
      zdim = 1;
    end

    sawe_plot_data(:,:,zdim) = [t f a];
end %for

end %function
