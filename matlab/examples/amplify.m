function [data]=amplify(data)

disp(['amplify ' sawe_getdatainfo(data)]);

data.samples = data.samples*4;

