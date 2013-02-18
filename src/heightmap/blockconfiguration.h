#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"

#include <boost/shared_ptr.hpp>

namespace Heightmap {

class Collection;

class BlockConfiguration {
public:
    typedef boost::shared_ptr<BlockConfiguration> Ptr;

    BlockConfiguration(Collection*);

    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    void samplesPerBlock(unsigned);
    void scalesPerBlock(unsigned);

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    // Requires Collection::transform.
    Tfr::FreqAxis transform_scale() const;
    float displayedTimeResolution(float ahz) const;

    // Requires Collection::target.
    float targetSampleRate() const;
    float length() const;

private:
    Collection* collection_;

    unsigned    scales_per_block_;
    unsigned    samples_per_block_;

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
