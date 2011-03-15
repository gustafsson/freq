function [data]=amplify2(data)

disp(['amplify2 ' sawe_getdatainfo(data)]);

% Amplify the first channel with a factor 4
data.buffer(:,1) = data.buffer(:,1)*4;
if size(data.buffer,2)>1
  % Amplify the second channel (if any) with a factor 0.5
  data.buffer(:,2) = data.buffer(:,2)*0.5;
end
% Leave any other channels unaffected


% Plot two lines
% from (1 s, 100 Hz, amplitude 0.5) to (2 s, 200 Hz, amplitude 1)
data.plot(:,:,1)=[1 100 0.5
                  2 200 1];
% from (1 s, 400 Hz, amplitude 2) to (2 s, 500 Hz, amplitude 0.4)
data.plot(:,:,2)=[1 400 2
                  2 500 0.4];

