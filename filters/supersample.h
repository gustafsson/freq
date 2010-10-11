#ifndef SUPERSAMPLE_H
#define SUPERSAMPLE_H

#include "signal/source.h"

namespace Filters {

class SuperSample
{
public:
    static Signal::pBuffer supersample( Signal::pBuffer b, float requested_sample_rate );
};

} // namespace Filters
#endif // SUPERSAMPLE_H
