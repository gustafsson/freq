#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "signal/intervals.h"

#include <boost/shared_ptr.hpp>

namespace Heightmap {

class BlockConfiguration {
public:
    typedef boost::shared_ptr<BlockConfiguration> Ptr;

    BlockConfiguration(float fs);

    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    void samplesPerBlock(unsigned);
    void scalesPerBlock(unsigned);

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    // targetSampleRate is used to compute which rawdata (Signal::Interval) that a block represents
    float targetSampleRate() const;

private:
    unsigned    scales_per_block_;
    unsigned    samples_per_block_;
    float       sample_rate;

    /**
      Heightmap blocks are rather agnostic to FreqAxis. But it's needed to create them.
      */
    Tfr::FreqAxis display_scale_;

    /**
      Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But it's needed to create them.
      */
    AmplitudeAxis amplitude_axis_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCONFIGURATION_H
