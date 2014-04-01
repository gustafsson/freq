#ifndef FILTERS_RIDGE_H
#define FILTERS_RIDGE_H

#include "tfr/cwtfilter.h"

namespace Filters
{
    class Ridge: public Tfr::CwtChunkFilter
    {
        void subchunk( Tfr::ChunkAndInverse& chunk );
    };


    class RidgeDesc: public Tfr::CwtChunkFilterDesc {
    public:
        Tfr::pChunkFilter       createChunkFilter(Signal::ComputingEngine* engine) const;
        ChunkFilterDesc::ptr    copy() const;
    };
}
#endif // FILTERS_RIDGE_H
