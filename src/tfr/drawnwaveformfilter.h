#ifndef DRAWNWAVEFORMFILTER_H
#define DRAWNWAVEFORMFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class DrawnWaveformFilter : public Tfr::Filter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    DrawnWaveformFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );


    /**
      Computes the interval that computeChunk would need to work.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );


    void applyFilter( ChunkAndInverse &chunk );

private:

    float max_value_;
};

} // namespace Tfr

#endif // DRAWNWAVEFORMFILTER_H
