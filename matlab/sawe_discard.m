% typical usage
%   sawe_discard(data)
%   sawe_discard(data, discardFront, discardBack)
%
% As default, discards samples from data.buffer as given by data.redundancy.
% Optionally some other value could be discarded from the front and back respectively
function data=sawe_discard(data, discardFront, discardBack)
if nargin<1
    error('Error');
elseif nargin<2
    discardBack = discardFront = data.overlap;
elseif nargin<3
    discardBack = discardFront;
end


if 0==data.offset
    data.samples = data.samples(1:end-discardBack,:);
else
    data.offset = data.offset + discardFront;
    data.samples = data.samples(1+discardFront:end-discardBack,:);
end

