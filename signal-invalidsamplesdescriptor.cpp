#include "signal-invalidsamplesdescriptor.h"
#include <stdexcept>

namespace Signal {

InvalidSamplesDescriptor::InvalidSamplesDescriptor()
{
}

InvalidSamplesDescriptor& InvalidSamplesDescriptor::operator |= (const InvalidSamplesDescriptor& /*b*/)
{
    throw std::logic_error("Not implemented");
}

} // namespace Signal
