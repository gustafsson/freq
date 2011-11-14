%
% sawe_plot, plots lines in Sonic AWE
%
%   sawe_plot(t, hz)
%
% 't' is a vector with coordinates along the time axis.
% 'hz' is either a vector of the same length as 't' or a scalar.
%
%
% To plot multiple lines with different coordinates along the time axis, call sawe_plot multiple times or group parameters in pairs:
%
%   sawe_plot(t1, hz1, t2, hz2, ... tn, hzn)
%
%
% See also sawe_plot2 which can plot lines with varying amplitude.
%
function sawe_plot(varargin);

    if (3==nargin)
	    sawe_plot2(varargin{:});
		return
	end
	
	default_args = cell(0);

	if 0~=mod(nargin,2)
		error('Input must be a multiple of 2');
	end

	for n=1:nargin/2

		t = varargin{2*(n-1) + 1};
		f = varargin{2*(n-1) + 2};
		a = 1;

		default_args{3*(n-1) + 1} = t;
		default_args{3*(n-1) + 2} = f;
		default_args{3*(n-1) + 3} = a;

	end %for

	sawe_plot2(default_args{:})

end %function
