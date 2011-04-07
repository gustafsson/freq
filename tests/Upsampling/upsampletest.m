% Random input signal
rand("seed",1);
y = rand(256,1);

selection=2:55;
fy = fft(y);


% Do this the normal way
fscale(1:length(fy)) = 0;
fscale(selection) = fy(selection)
scale = ifft(fscale);

fscale2(1:length(fy)/2) = 0;
fscale2(selection) = fy(selection)
scale2 = ifft(fscale2);
scale2 = real(scale2);

% upsample
upft = fft(scale2);
upft(1) = upft(1)/2;
upft(65) = upft(65)/2;
upft(66:256) = 0;
upsamples_scale = ifft(upft);
max(abs(real(upsamples_scale)-scale))

