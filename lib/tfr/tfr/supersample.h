#ifndef SUPERSAMPLE_H
#define SUPERSAMPLE_H

#include "signal/source.h"
#include "tfrdll.h"

namespace Tfr {

class TfrDll SuperSample
{
public:
    static Signal::pMonoBuffer supersample( Signal::pMonoBuffer b, float requested_sample_rate );
};

} // namespace Tfr
#endif // SUPERSAMPLE_H
