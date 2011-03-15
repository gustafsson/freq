function [data]=fantracker(data, args)

windowsize = 2^11;
max_hz = 70;
min_hz = 50;
channelnumber = 1:size(data.buffer,2);

if nargin==2 && ~isempty(args)
  windowsize = args(1);
  if (numel(args)>1);  min_hz = args(2); end
  if (numel(args)>2);  max_hz = args(3); end
  if (numel(args)>3);  channelnumber = args(4); end
end

min_hz = max(min_hz, 2*data.samplerate/windowsize);

disp( ['fantracker ' getdatainfo(data) ' windowsize=' num2str(windowsize) ' hz=[' num2str(min_hz) ', ' num2str(max_hz) ']' ] );

global state;
if data.offset==0
  state.hz = [];
  state.e = 0.3;
end

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

for channel=channelnumber
  signal = data.buffer(:, channel);

  [ft, dummy] = stft(signal, windowsize, windowsize, windowsize/2, 3);
  C = log(4 + abs(ft));
  [C, dummy] = stft(C(:), windowsize, windowsize, windowsize/2, 3);

  clip_index = max(1, round(data.samplerate/max_hz));
  C(1:clip_index, :) = 0;
  clip_index = max(1, round(data.samplerate/min_hz));
  clip_index = min(clip_index, windowsize/2);
  C(clip_index:end, :) = 0;

  [a,j]=max(abs(C));

  hz = data.samplerate ./ j;
  x = linspace(data.offset + windowsize/2, data.offset + size(data.buffer,1) - windowsize/2, numel(hz));
  t = x / data.samplerate;
  a = max(0.3, min(1, 100*a ./ sum(abs(C))));

  % Plot which peak that was found
  %data.plot(:,:,channel*2 ) = [t' hz' a'];

  % For each window
  hz2 = zeros(size(hz));
  for k=1:numel(hz)
    if isempty(state.hz)
      state.hz = hz(1);
    else
      state.hz = state.hz*0.95 + hz(k)*0.05;
    end
    hz2(k) = state.hz;
  end

  % Fetch overtone number 4 and 8
  overtones = [4 8];

  % Indicies in fourier domain
  k = windowsize*hz/data.samplerate;
  n = overtones'*k;

  y = abs(ft(:));
  offs = ones(1,size(n,1))'*(0:size(ft,2)-1)*size(ft,1);
  m = round(n);
  x = n - m;
  y1 = y(m-1+offs);
  y2 = y(m+offs);
  y3 = y(m+1+offs);
  
  %solve
  %y1 = k+p+q;
  %y2 = q;
  %y3 = k-p+q;

  p=(y1-y3)./2;
  q=y2;
  k=y1 + y3 - 2*q;

  pattern = k.*x.*x + p.*k + q; 
%  pattern = (1-b).*y(m + offs) + b.*y(m + offs + 1);

  % Plot where pattern is changing the most
%  patterndiff = pattern(:, 2:end)-pattern(:, 1:end-1);
%  patterndiff(:,end+1) = patterndiff(:,end);
%  [v,m]=min(pattern);
  v=min(pattern);
%  hz2 = 2.*k/windowsize*data.samplerate;
  a = zeros(size(a));
  % For each window
  for k=1:numel(hz)
    state.e = state.e*0.95;
    if v<0.5
      state.e = min(2, state.e + 0.2);
    end

    a(k) = state.e > 0.5;
  end

  data.plot(:,:,channel*2-1 ) = [t' hz2' a'];
%  data.plot(:,:,channel*2-1 ) = [t' hz2' v'];
end

