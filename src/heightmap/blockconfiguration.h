#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "signal/intervals.h"

#include <boost/shared_ptr.hpp>

namespace Heightmap {

class BlockSize {
public:
    BlockSize(int texels_per_row, int texels_per_column);

    int texels_per_row() const;
    int texels_per_column() const;
    int texels_per_block() const { return texels_per_row() * texels_per_column(); }

private:
    int texels_per_column_;
    int texels_per_row_;
};


class BlockConfiguration {
public:
    // TODO remove Ptr
    typedef boost::shared_ptr<BlockConfiguration> Ptr;

    BlockConfiguration(BlockSize block_size, float fs);

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

#endif // HEIGHTMAP_BLOCKCONFIGURATION_H
