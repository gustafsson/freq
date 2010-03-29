#include "signal-operation.h"

namespace Signal {

unsigned Operation::
sample_rate() const
{
    return _child->sample_rate();
}

unsigned Operation::
number_of_samples() const
{
    return _child->number_of_samples();
}

InvalidSamplesDescriptor Operation::
updateIsd()
{
    _isd |= _child->updateIsd();
    return _isd;
}

} // namespace Signal
