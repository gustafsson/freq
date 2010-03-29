#include "signal-invalidsamplesdescriptor.h"

namespace Signal {

InvalidSamplesDescriptor::InvalidSamplesDescriptor()
{
}

InvalidSamplesDescriptor& operator |= (const InvalidSamplesDescriptor& b)
{
    throw "Not implemented";
}

} // namespace Signal
