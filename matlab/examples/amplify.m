function [data]=amplify(data)

disp(['amplify ' sawe_getdatainfo(data)]);

data.buffer = data.buffer*4;

