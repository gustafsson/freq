spectrum = zeros(7,4);
spectrum(1,:) = [ 1, 0, 0, 0 ];
spectrum(2,:) = [ 0, 0, 1, 0 ];
spectrum(3,:) = [ 0, 1, 1, 0 ];
spectrum(4,:) = [ 0, 1, 0, 0 ];
spectrum(5,:) = [ 1, 1, 0, 0 ];
spectrum(6,:) = [ 1, 0, 1, 0 ];
spectrum(7,:) = [ 1, 0, 0, 0 ];

count = 6;
wavelengthScalar = 0:0.01:1;
f = count*wavelengthScalar;
i1 = floor(max(0.0, min(f-1.0, count)));
i2 = floor(min(f, count));
i3 = floor(min(f+1.0, count));
i4 = floor(min(f+2.0, count));
s = 0.5+(f-i2)*0.5;
t = (f-i2)*0.5;

rgb = zeros(4,numel(wavelengthScalar));
for k=1:4
p = spectrum(:,k);
c = p(1+i1)'.*(1-s) + p(1+i3)'.*s + p(1+i2)'.*(1-t) + p(1+i4)'.*t;
x = 1-(1-wavelengthScalar).*(1-wavelengthScalar).*(1-wavelengthScalar);
c = 1.*(1-x) + min(0.7,c.*0.5).*x;
rgb(k,:) = c;

end

plot(rgb');
