#ifndef SUPERSAMPLE_H
#define SUPERSAMPLE_H

#include "signal/source.h"

namespace Tfr {

class SaweDll SuperSample
{
public:
    static Signal::pMonoBuffer supersample( Signal::pMonoBuffer b, float requested_sample_rate );
};

} // namespace Tfr
#endif // SUPERSAMPLE_H
