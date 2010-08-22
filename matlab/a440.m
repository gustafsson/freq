fs=44100;
t=0:1/fs:1;
hz=440;
y=sin(hz*2*pi*t');
N=floor(numel(y)/4);
s=linspace(0,1,N)';
s=[s; ones(numel(y)-2*N, 1); flipud(s)];
wavwrite(.5*y.*s,fs,"a440.wav");
