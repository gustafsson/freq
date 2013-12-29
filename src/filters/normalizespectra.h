#ifndef FILTERS_NORMALIZESPECTRA_H
#define FILTERS_NORMALIZESPECTRA_H

#include "tfr/stftfilter.h"

namespace Filters {

class NormalizeSpectraKernel : public Tfr::ChunkFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    // negative values set a fraction rather than an absolute number of Hz
    NormalizeSpectraKernel(float meansHz=0.1f);

    void operator()( Tfr::ChunkAndInverse& chunk );

private:
    float meansHz_;

    void removeSlidingMean( Tfr::Chunk& chunk );
    void removeSlidingMedian( Tfr::Chunk& chunk );

    int computeR( const Tfr::Chunk& chunk );
};


class NormalizeSpectra : public Tfr::StftFilterDesc
{
public:
    NormalizeSpectra(float meansHz=0.1f);
};

} // namespace Filters

#endif // FILTERS_NORMALIZESPECTRA_H
