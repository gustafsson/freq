fs=44100;
t=0:1/fs:1;
hz=440;
y=sin(hz*2*pi*t');
wavwrite(y,fs,"a440.wav");
