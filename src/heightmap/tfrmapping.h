#ifndef HEIGHTMAP_TFRMAPPING_H
#define HEIGHTMAP_TFRMAPPING_H

#include "blocksize.h"
#include "tfr/transform.h"
#include "volatileptr.h"
#include "signal/poperation.h"

#include <vector>

namespace Heightmap {
class Collection;

class TfrMapping {
public:
    TfrMapping(BlockSize, float fs);

    bool operator==(const TfrMapping& b);

    BlockSize               block_size;
    float                   targetSampleRate;
    float                   length;

    /**
     * Not that this is the transform that should be used. Blocks computed by
     * an old transform desc might still exist as they are being processed.
     */
    Tfr::TransformDesc::Ptr transform_desc;

    /**
     * Heightmap blocks are rather agnostic to FreqAxis. But it's needed to
     * create them.
     */
    Tfr::FreqAxis display_scale;

    /**
     * Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But
     * it's needed to create them.
     */
    AmplitudeAxis amplitude_axis;
};


class TfrMap: public VolatilePtr<TfrMap> {
public:
    TfrMap(TfrMapping tfr_mapping, int channels);
    ~TfrMap();

    BlockSize block_size() const;
    void block_size(BlockSize bs);

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    // targetSampleRate is used to compute which rawdata (Signal::Interval) that a block represents
    float targetSampleRate() const;

    Tfr::TransformDesc::Ptr transform_desc() const;
    void transform_desc(Tfr::TransformDesc::Ptr);

    const TfrMapping& tfr_mapping() const;

    float length() const;
    void length(float L);

    int channels() const;

    typedef VolatilePtr<Heightmap::Collection>::Ptr pCollection;
    typedef std::vector<pCollection> Collections;
    Collections collections() const;

private:
    void updateCollections();

    TfrMapping  tfr_mapping_;
    Collections collections_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPING_H
