% This file is distributed for evaluation purposes only
% Author: johan.gustafsson@sonicawe.com
function data=markclicks(data, limits)

disp(['markclicks ' sawe_getdatainfo(data)]);

%% Validate input arguments
derivatives=5;
if nargin==1 || isempty(limits)
  limits = 0.005;
end

if 1==numel(limits)
  limits = ones(derivatives, 1)*limits;
elseif numel(limits)~=derivatives
  derivatives = numel(limits);
end

global foundclicks
if isempty(foundclicks)
  foundclicks = 0;
end

%% Compute 'derivatives' number of derivatives
p = mean(data.buffer, 2);
d = zeros(numel(p), derivatives);
for k=1:derivatives
   p = conv(p, [1, -1]);
   d(:,k) = p((1:size(data.buffer,1))+mod(k,2));
end 


%% Discard invalid data after convolution
% The plotting below needs +1 here
data.redundancy = ceil(derivatives/2) + 1;
data = sawe_discard(data);
if 0==data.offset
    d=d(1:end-data.redundancy,:);
else
    d=d(1+data.redundancy:end-data.redundancy,:);
end


%% Plot the points where the derivative is above a given limit
x = 1:size(data.buffer,1);
t = (data.offset + x) / data.samplerate;
d = abs(d);

clicktest = zeros(size(d));
hz = logspace(log10(data.samplerate/20), log10(data.samplerate/3), derivatives+1);

for k=1:derivatives
   p = d(:,k);
   n = p>limits(k);

   clicks = [0; (n(2:end-1) & ~n(1:end-2) & ~n(3:end)); 0];
   clicktest(:,k) = clicks;

   m = conv(n, [1 1 1]);
   m = 0~=m(2:end-1);
   sawe_plot2(t(m), hz(k), n(m));

   m = conv(clicks, [1 1 1]);
   m = 0~=m(2:end-1);
   sawe_plot2(t(m), hz(k)*0.9 + 0.1*hz(k+1), n(m));
end

foundclicks = foundclicks + sum(any(clicktest,2));

if all(limits==limits(1))
  limitstr = ['threshold ' num2str(limits(1))];
else
  limitstr = ['thresholds [' num2str(limits') '] respectively'];
end

disp(['Found ' num2str(foundclicks') ' clicks so far using the 1st, 2nd, ... derivative and ' limitstr]);

