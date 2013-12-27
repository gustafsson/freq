#ifndef FILTERS_REASSIGN_H
#define FILTERS_REASSIGN_H

#include "tfr/cwtfilter.h"

namespace Filters
{
    class Reassign: public Tfr::CwtFilter
    {
    public:
        virtual void operator()( Tfr::Chunk& chunk );

        void limitedCpu(Tfr::Chunk& chunk );
        void naiveCpu(Tfr::Chunk& chunk );
        void brokenGpu(Tfr::Chunk& chunk );
    };


    class Tonalize: public Tfr::CwtFilter
    {
    public:
        virtual void operator()( Tfr::Chunk& );

        void brokenGpu(Tfr::Chunk& chunk );
    };

}
#endif // FILTERS_REASSIGN_H
