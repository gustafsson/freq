#ifndef FILTERS_NORMALIZESPECTRA_H
#define FILTERS_NORMALIZESPECTRA_H

#include "tfr/stftfilter.h"

namespace Filters {

class NormalizeSpectra : public Tfr::ChunkFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    // negative values set a fraction rather than an absolute number of Hz
    NormalizeSpectra(float meansHz=0.1f);

    void operator()( Tfr::ChunkAndInverse& chunk );

private:
    float meansHz_;

    void removeSlidingMean( Tfr::Chunk& chunk );
    void removeSlidingMedian( Tfr::Chunk& chunk );

    int computeR( const Tfr::Chunk& chunk );
};


class NormalizeSpectraDesc : public Tfr::StftFilterDesc
{
public:
    NormalizeSpectraDesc(float meansHz=0.1f);
    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;

private:
    float meansHz;
};

} // namespace Filters

#endif // FILTERS_NORMALIZESPECTRA_H
