function [data]=verifyregulation(data)


% Specify 5 bandpass windows (0-100), (100-1000), etc.
hz1 = [ 0,   100,  1000, 4000,  50 ];
hz2 = [ 100, 1000, 4000, 10000, 22050 ];

% Limits for each of the 5 windows
limits_db = [ -10, -20, -20, 20, -20 ];
p0 = 1; % reference RMS level for 0 dB


% Specify window size for the transform
windowsize = 2^12; % 8192 samples -> 4096 different frequencies
increment = 2^11;    % 2048 samples
numcoeff = 2^12;
windowtype = 1; % hanning window, other window types are easily available

%windowsize = 2^13;
%increment = 2^13;
%numcoeff = 2^12;
%windowtype = 3; % regular window

data.redundancy = windowsize - increment;

% Align input
align = numcoeff*2;
removefromstart = ceil(data.offset/align)*align-data.offset;
removefromend = mod(data.offset+size(data.buffer,1), align);
data.offset = data.offset + removefromstart;
data.buffer = data.buffer(1+removefromstart:end-removefromend,:);

% Compute power levels for the different bandpass windows
% Bandpass with high q-factor in this simple example
mono = sum(data.buffer, 2);
[Y, C] = stft(mono, windowsize, increment, numcoeff, windowtype);

if isempty(Y)
  data.buffer = [];
  return
end

% Display some information on this block of data
%disp(['verifyregulation ' sawe_getdatainfo(data)]);

x = 1:size(Y,2);
t = data.offset/data.samplerate + (x'-0.5) / data.samplerate*windowsize;
signals=zeros(increment*size(Y,2), numel(hz1));

for k=1:numel(hz1)
  l1 = round(hz1(k)*align/data.samplerate);
  l2 = round(hz2(k)*align/data.samplerate);
  bandpass = Y;
  bandpass([1+(0:l1-1) (l2:align-l2) align-(0:l1-1)], :) = 0;
  s = synthesis(bandpass, C);
  signals(:,k) = s;
end

p = zeros(size(Y,2), numel(hz1));
for k=1:numel(hz1)
  s = reshape(signals(:,k), size(signals,1)/size(p,1), size(p,1));
  p(:,k) = sqrt(mean(s .* conj(s)));
end
dB = 10*log10(p/p0);
limits_p = 10.^(limits_db/10)*p0;
format short
for k=numel(hz1):-1:1
	sawe_plot(t, mean([hz1(k) hz2(k)]), p(:,k)/limits_p(k));
    v = p(:,k)>limits_p(k);
    lhz = ones(2,numel(v));
    lhz(1,:) = hz1(k);
    lhz(2,:) = hz2(k);
    td = ones(2,1)*t';
    td(2,:) = td(2,:) + 0.0001;
    v = ones(2,1)*v';
    v(v==0)=NaN;
    v = v*(1+0.16*log10(hz1(k)/hz2(k)));
	sawe_plot(td, lhz, v);
end

% Display results
disp(['Bandpassed dB levels in the interval ' num2str(data.offset/data.samplerate) '-' num2str((data.offset + numel(mono))/data.samplerate) ' s']);
for k=1:numel(hz1)
    if mean(dB(:,k))>limits_db(k)
      disp([ num2str(hz1(k)) '-' num2str(hz2(k)) ' Hz: ' num2str(mean(dB(:,k))) ' dB > ' num2str(limits_db(k)) '!']);
    else
      disp([ num2str(hz1(k)) '-' num2str(hz2(k)) ' Hz: ' num2str(mean(dB(:,k))) ' dB']);
    end
end

