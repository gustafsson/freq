function [data]=printpeaks(data)

F = fft(sum(data.samples,2));
F(1) = 0; % skip DC component
F = F(1:end/2); % skip redundant part of spectrum
[val, j] = max(abs(F)); 

% Make 'j' more accurate by a quadratic interpolation
if j~=1 && j~=length(F)
	% y(x) = k*x*x + p*x + v(2)
	% y(-1) = v(1), y(0) = v(2) and y(1) = v(3)
	v = abs(F(j-1:j+1));
    k = (v(1) - 2*v(2) + v(3))/2;
    p = (-v(1) + v(3))/2;
	% y has a maxima at x0 given by:
    x0 = -p/(2*k);
	j = j + x0;
end

hz = (j-1) * data.fs/size(data.samples,1);
dhz = data.fs/size(data.samples,1);
dhz = sqrt(1/12)*dhz;

plusminus = char([0xB1]);
disp(['[' sprintf('%.2f', data.offset/data.fs) ', ' sprintf('%.2f', (data.offset+size(data.samples,1))/data.fs) ')' ...
	  ' s, peak ' sprintf('%.2f', hz) ' ' plusminus ' ' sprintf('%.2f', dhz) ' Hz']);

