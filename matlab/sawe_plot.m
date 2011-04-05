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
