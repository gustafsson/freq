#ifndef FILTERS_REASSIGN_H
#define FILTERS_REASSIGN_H

#include "tfr/cwtfilter.h"

namespace Filters
{
    class Reassign: public Tfr::ChunkFilter
    {
    public:
        void operator()( Tfr::ChunkAndInverse& chunk );

        void limitedCpu(Tfr::Chunk& chunk );
        void naiveCpu(Tfr::Chunk& chunk );
        void brokenGpu(Tfr::Chunk& chunk );
    };


    class Tonalize: public Tfr::ChunkFilter
    {
    public:
        void operator()( Tfr::ChunkAndInverse& chunk );

        void brokenGpu(Tfr::Chunk& chunk );
    };


    class ReassignDesc: public Tfr::CwtChunkFilterDesc {
    public:
        Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const
        {
            return Tfr::pChunkFilter(new Reassign);
        }
    };


    class TonalizeDesc: public Tfr::CwtChunkFilterDesc {
    public:
        Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine=0) const
        {
            return Tfr::pChunkFilter(new Tonalize);
        }
    };

}
#endif // FILTERS_REASSIGN_H
