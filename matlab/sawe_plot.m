function sawe_plot(varargin);

	default_args = cell;

	assert( ~mod(nargin,2), 'Input must be a multiple of 2');

	for n=1:nargin/2

		t = varargin{2*(n-1) + 1};
		f = varargin{2*(n-1) + 2};
		a = 1;

		default_args{3*(n-1) + 1} = t;
		default_args{3*(n-1) + 2} = f;
		default_args{3*(n-1) + 3} = a;

	endfor

	sawe_plot2(default_args{:})

endfunction
