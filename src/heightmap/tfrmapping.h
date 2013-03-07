#ifndef HEIGHTMAP_TFRMAPPING_H
#define HEIGHTMAP_TFRMAPPING_H

#include "blocksize.h"

namespace Heightmap {

class TfrMapping {
public:
    // TODO remove Ptr
    typedef boost::shared_ptr<TfrMapping> Ptr;

    TfrMapping(BlockSize block_size, float fs);

    int samplesPerBlock() const;
    int scalesPerBlock() const;
    BlockSize block_size() const;

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    // targetSampleRate is used to compute which rawdata (Signal::Interval) that a block represents
    float targetSampleRate() const;

private:
    BlockSize       block_size_;
    float           sample_rate_;

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

#endif // HEIGHTMAP_TFRMAPPING_H
