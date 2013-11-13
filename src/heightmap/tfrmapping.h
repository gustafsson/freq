#ifndef HEIGHTMAP_TFRMAPPING_H
#define HEIGHTMAP_TFRMAPPING_H

#include "blocklayout.h"
#include "volatileptr.h"
#include "signal/poperation.h"
#include "visualizationparams.h"

#include <vector>

namespace Heightmap {
class Collection;
typedef int ChannelCount;

class TfrMapping: public VolatilePtr<TfrMapping> {
public:
    TfrMapping(BlockLayout, ChannelCount channels);
    ~TfrMapping();

    BlockLayout block_layout() const;
    void block_layout(BlockLayout bs);

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    float targetSampleRate() const;
    void targetSampleRate(float);

    Tfr::TransformDesc::Ptr transform_desc() const;
    void transform_desc(Tfr::TransformDesc::Ptr);

    float length() const;
    void length(float L);

    int channels() const;
    void channels(int value);

    typedef VolatilePtr<Heightmap::Collection>::Ptr pCollection;
    typedef std::vector<pCollection> Collections;
    Collections collections() const;

private:
    void updateCollections();

    Collections                 collections_;
    BlockLayout                 block_layout_;
    VisualizationParams::Ptr    visualization_params_;
    float                       length_;

public:
    static void test();
    static TfrMapping::Ptr testInstance();
};

} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPING_H
