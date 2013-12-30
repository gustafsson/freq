#ifndef FILTERS_RIDGE_H
#define FILTERS_RIDGE_H

#include "tfr/cwtfilter.h"

namespace Filters
{
    class Ridge: public Tfr::ChunkFilter
    {
        void operator()( Tfr::ChunkAndInverse& chunk );
    };


    class RidgeDesc: public Tfr::CwtFilterDesc {
    public:
        RidgeDesc():Tfr::CwtFilterDesc(Tfr::pChunkFilter(new Ridge)){}
    };
}
#endif // FILTERS_RIDGE_H
