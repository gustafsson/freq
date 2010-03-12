function T=plot_producevideo(n)

if nargin~=1 || isscalar(n) || ~ismatrix(n)
	T=read_csv_from_sonicawe(n);
else
	T=n;
end

wavelet_std_t=0.06;
scales_per_sample=40;
fs=44100;
HI = max(max(abs(T)));
frames = 400;
k=1;

while exist(["image-" num2str(k) ".png"])
	k=k+1;
end

if k==1
	first = 1;
	last = size(T,2)-wavelet_std_t*fs*2;
else
	first = wavelet_std_t*fs;
	last = size(T,2)-first;
end

for n=first:ceil((last-frames)/frames):last;
	spectra = T(:,n);
	spectra2d = zeros(scales_per_sample,ceil(numel(spectra)/scales_per_sample));
	spectra2d(1:numel(spectra))=spectra;
	imagesc(abs(spectra2d),[0,HI]);
	print(["image-" num2str(k) ".png"]);
	k=k+1;
end

