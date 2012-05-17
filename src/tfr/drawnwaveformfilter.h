#ifndef DRAWNWAVEFORMFILTER_H
#define DRAWNWAVEFORMFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class DrawnWaveformFilter : public Tfr::Filter
{
public:
    DrawnWaveformFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );


    /**
      Computes the interval that computeChunk would need to work.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I );


    /**
      This computes the Stft chunk covering a given interval.
      */
    ChunkAndInverse computeChunk( const Signal::Interval& I );


    /**
      Get the Tfr::Transform for this operation.
      */
    Tfr::pTransform transform() const;


    /**
      Set the Tfr::Transform for this operation and update _invalid_samples.
      Will throw throw std::invalid_argument if 'm' is not an instance of
      DrawnWaveform.
      */
    void transform( Tfr::pTransform m );
};

} // namespace Tfr

#endif // DRAWNWAVEFORMFILTER_H
