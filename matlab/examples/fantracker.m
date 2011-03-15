function [data]=fantracker(data, windowsize)

if nargin==1 || isempty(windowsize)
    windowsize = 2^11;
end

disp( ['fantracker ' getdatainfo(data) ' windowsize=' num2str(windowsize)] );

% Make sure data is aligned to windowsize
start = mod(windowsize - mod(data.offset,windowsize), windowsize);
stop = start + floor((size(data.buffer,1)-start)/windowsize)*windowsize;

data.buffer = data.buffer(1+start:stop, :);
data.offset = data.offset + start;

if 0==numel(data.buffer)
  disp(['data.buffer must be at least ' num2str(windowsize) ' samples long']);
  data.redundancy = windowsize;
  return
end

data.redundancy = 0;

for channel=1:size(data.buffer,2)
  signal = data.buffer(:, channel);

  [ft, dummy] = stft(signal, windowsize, windowsize, windowsize/2, 3);
  C = log(4 + abs(ft));
  [C, dummy] = stft(C(:), windowsize, windowsize, windowsize/2, 3);

  max_hz = 100;
  min_hz = 30;
  min_hz = max(min_hz, 2*data.samplerate/windowsize);
  clip_index = round(data.samplerate/max_hz);
  C(1:clip_index, :) = 0;
  clip_index = round(data.samplerate/min_hz);
  clip_index = min(clip_index, windowsize/2);
  C(clip_index:end, :) = 0;

  [a,j]=max(abs(C));

  hz = data.samplerate ./ j;
  x = linspace(data.offset + windowsize/2, data.offset + size(data.buffer,1) - windowsize/2, numel(hz));
  t = x / data.samplerate;
  a = min(1, a * (200/windowsize));
  %a = min(1, 2*a ./ sum(abs(C)));

  % Plot which peak that was found
  %%data.plot(:,:,channel*2 -1 ) = [t' hz' a'];
  data.plot(:,:,channel ) = [t' hz' a'];

  % Fetch pattern
  % Indicies in fourier domain
%  k = windowsize*hz/data.samplerate;

%  n = round([1:10]'*k);
%hz

%  y = ft(:);
%  offs = ones(1,size(n,1))'*(0:size(ft,2)-1)*size(ft,1);
%  pattern = y(n + offs);

  % Plot if pattern is changing
%  patterndiff = pattern(:, 2:end)-pattern(:, 1:end-1);
%  patterndiff(:,end+1) = patterndiff(:,end);
%  [v,m]=max(patterndiff);
%  hz2 = m/windowsize*data.samplerate;

%hz2

 % data.plot(:,:,channel*2 +1 ) = [t' hz2' v'];
end
