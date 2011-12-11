y=rand(16,1);
windowsize=4;
inc=2;
prepy=zeros(windowsize, size(y,1)/inc-1);

% rectangular window
for n=0:size(prepy,2)-1
  s = n*inc;
  prepy(:,1+n)=y(1+s:s+windowsize);
  % apply window function here
end

f = fft(prepy);
f2 = ifft(f);

% inverse
y2=zeros(inc*(size(f2,2)-1)+windowsize,1);
for n=0:size(f2,2)-1
  s = n*inc;
  y2(1+s:s+windowsize) = y2(1+s:s+windowsize) + f2(:,1+n);
end

y
f
y2*inc/windowsize
success = all(abs(y(3:end-2)-y2(3:end-2)*inc/windowsize)<0.0001)
