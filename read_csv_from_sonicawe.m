function T = read_csv_from_sonicawe(n)

if nargin~=1
	error "Provide csv file to read from. Either in text or a number such as '1' which will be interpreted as 'sonicawe-1.csv'"
end

filename = n;

if isscalar(n)
	filename = ['sonicawe-' num2str(n) '.csv'];
end

T1 = load ('-ascii',filename);

T = T1(:,1:2:end) + i*T1(:,2:2:end);
