function T=plot_producevideo(n,scales_per_sample,framespath)

if nargin<1 || isscalar(n) || ~ismatrix(n) || 0==numel(n)
	T=read_csv_from_sonicawe(n);
else
	T=n;
end

if nargin<2 || 0==numel(scales_per_sample)
	scales_per_sample=40;
end

if nargin<3 || 0==numel(framespath)
	framespath = "frames";
end

wavelet_std_t=0.06;
fs=44100;
HI = max(max(abs(T)));
frames = 400;

[r,k]=system(["ls \"" framespath "\"/image-*.png | grep -c ."]);
k=str2num(k)+1;

if k==1
	first = 1;
	last = size(T,2)-wavelet_std_t*fs*2;
else
	first = wavelet_std_t*fs;
	last = size(T,2)-first;
end

for n=first:ceil((last-first)/frames):last;
	spectra = T(:,n);
	spectra2d = zeros(scales_per_sample,ceil(numel(spectra)/scales_per_sample));
	spectra2d(1:numel(spectra))=spectra;
	imagesc(abs(spectra2d),[0,HI]);
	print([framespath "/image-" num2str(k) ".png"]);
	k=k+1;
end

