function T=plotspectra2d(n)

if nargin~=1 || isscalar(n) || ~ismatrix(n)
	T=read_csv_from_sonicawe(n);
else
	T=n;
end

spectra = T(:,round(end/2));
scales_per_sample=40;
spectra2d = zeros(scales_per_sample,ceil(numel(spectra)/scales_per_sample));
spectra2d(1:numel(spectra))=spectra;
figure(1);
plot(abs(spectra));
figure(2);
imagesc(abs(spectra2d));

