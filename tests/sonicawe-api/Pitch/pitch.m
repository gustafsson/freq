[y,fs]=wavread('pling.wav');
f=fft(y);
f(end/2:end,:)=0;
y2=ifft(f);
t=(0:length(y)-1)/fs;
# move entire spectrum 200 hz, will mirror in fs/2
dhz = 200;
y3=y2(:,1).*exp(i*t*2*pi*dhz);
wavwrite('pling2.wav', real(y3), fs);

