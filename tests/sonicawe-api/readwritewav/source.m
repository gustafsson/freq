fs=44100;
N=1024*1024*10;
y=rand(N,1);
wavwrite(y,fs,'source.wav');

