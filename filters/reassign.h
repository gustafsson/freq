#ifndef FILTERS_REASSIGN_H
#define FILTERS_REASSIGN_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"

namespace Filters
{
    class Reassign: public Tfr::CwtFilter
    {
        virtual void operator()( Tfr::Chunk& chunk );

        void limitedCpu(Tfr::Chunk& chunk );
        void naiveCpu(Tfr::Chunk& chunk );
    };
}
#endif // FILTERS_REASSIGN_H
