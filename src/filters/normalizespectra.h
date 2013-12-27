#ifndef FILTERS_NORMALIZESPECTRA_H
#define FILTERS_NORMALIZESPECTRA_H

#include "tfr/stftfilter.h"

namespace Filters {

class NormalizeSpectra : public Tfr::StftFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    // negative values set a fraction rather than an absolute number of Hz
    NormalizeSpectra(float meansHz=0.1f);

    virtual void operator()( Tfr::Chunk& );

private:
    float meansHz_;

    void removeSlidingMean( Tfr::Chunk& chunk );
    void removeSlidingMedian( Tfr::Chunk& chunk );

    int computeR( const Tfr::Chunk& chunk );
};

} // namespace Filters

#endif // FILTERS_NORMALIZESPECTRA_H
