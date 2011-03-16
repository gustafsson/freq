function sawe_plot2(varargin);

global sawe_hold_plot;
global sawe_plot_data;

assert(~mod(nargin,3),'Number of inputs must be a multiple of 3.');

offset = size(sawe_plot_data,3);
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
    endif

	% workaround to cope with lines of different length: repeat the last point to fill the vector!
	% this will not cause problems, as sawe will prune those points anyway.

    if max_length < length_ && n > 1

      pad_t = sawe_plot_data(end,1)*ones(length_ - max_length,1);
      pad_f = sawe_plot_data(end,2)*ones(length_ - max_length,1);
      pad_a = sawe_plot_data(end,3)*ones(length_ - max_length,1);

      sawe_plot_data = [sawe_plot_data; pad_t pad_f pad_a];

    endif

    max_length = max(max_length,length_);

    if (length_ < max_length)

      t = [t; t(end)*ones( max_length - length_, 1)];
      f = [f; f(end)*ones( max_length - length_, 1)];
      a = [a; a(end)*ones( max_length - length_, 1)];

    endif
    
		sawe_plot_data(:,:,offset+n) = [t f a];

endfor

endfunction
