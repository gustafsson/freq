#ifndef HEIGHTMAP_TFRMAPPING_H
#define HEIGHTMAP_TFRMAPPING_H

#include "heightmap/blocklayout.h"
#include "heightmap/visualizationparams.h"

#include "shared_state.h"
#include "shared_state_traits_backtrace.h"

#include "tfr/transform.h"

#include <vector>

namespace Heightmap {
class Collection;
typedef int ChannelCount;

class TransformDetailInfo : public DetailInfo {
public:
    TransformDetailInfo(Tfr::TransformDesc::ptr p);

    bool operator==(const DetailInfo&) const override;
    float displayedTimeResolution( float FS, float hz ) const override;
    float displayedFrequencyResolution( float FS, float hz1, float hz2 ) const override;

    Tfr::TransformDesc::ptr transform_desc() const { return p_; }

private:
    Tfr::TransformDesc::ptr p_;
};


class TfrMapping {
public:
    typedef shared_state<TfrMapping> ptr;
    typedef shared_state<const TfrMapping> const_ptr;
    typedef shared_state_traits_backtrace shared_state_traits;

    TfrMapping(BlockLayout, ChannelCount channels);
    ~TfrMapping();

    BlockLayout block_layout() const;
    void block_layout(BlockLayout bs);

    FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    float targetSampleRate() const;
    void targetSampleRate(float);

    Tfr::TransformDesc::ptr transform_desc() const;
    void transform_desc(Tfr::TransformDesc::ptr);

    double length() const;
    Signal::IntervalType lengthSamples() const;
    void lengthSamples(Signal::IntervalType L);

    int channels() const;
    void channels(int value);

    typedef shared_state<Heightmap::Collection> pCollection;
    typedef std::vector<pCollection> Collections;
    Collections collections() const;

    void gc();
private:
    void updateCollections();

    Collections                 collections_;
    Collections                 old_collections_;
    BlockLayout                 block_layout_;
    VisualizationParams::ptr    visualization_params_;
    Signal::IntervalType        length_samples_;

public:
    static void test();
    static TfrMapping::ptr testInstance();
};

} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPING_H
