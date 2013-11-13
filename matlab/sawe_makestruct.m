%
% Makes an instance that looks like the data structure used to pass data from Sonic AWE to scripts.
%
%   sawe_makestruct( samples )
%   sawe_makestruct( samples, fs )
%   sawe_makestruct( samples, fs, offset )
%   sawe_makestruct( samples, fs, offset, overlap )
%
%   samples
%     The actual samples to process. Each column is one channel.
%
%   fs
%     The number of samples per unit of time. Defaults to 44100.
%
%   offset
%     If the samples represent some part of the whole data. Defaults to 0.
%
%   overlap
%     Number of samples to be discarded before returning. Defaults to 0.
%
%
function [data]=sawe_makestruct(samples, fs, offset, overlap)

if nargin<4
  overlap = 0;
end
if nargin<3
  offset = 0;
end
if nargin<2
  fs = 44100;
end
if nargin<1
  error('syntax: sawe_makedata( samples ). Arange samples with channels i columns.')
end

data = struct();
data.samples = samples;
data.fs = fs;
data.offset = offset;
data.overlap = overlap;

