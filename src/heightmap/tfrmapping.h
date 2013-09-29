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


class TfrMapping {
public:
    TfrMapping(BlockLayout);

    bool operator==(const TfrMapping& b);
    bool operator!=(const TfrMapping& b);

    BlockLayout               block_layout;

    VisualizationParams::Ptr  visualization_params() const;
    BlockLayout               block_size() const;
    float                     targetSampleRate() const;
    Tfr::FreqAxis             display_scale() const;
    AmplitudeAxis             amplitude_axis() const;
    Tfr::TransformDesc::Ptr   transform_desc() const;

private:
    VisualizationParams::Ptr  visualization_params_;
};


class TfrMap: public VolatilePtr<TfrMap> {
public:
    TfrMap(TfrMapping tfr_mapping, ChannelCount channels);
    ~TfrMap();

    BlockLayout block_size() const;
    void block_size(BlockLayout bs);

    Tfr::FreqAxis display_scale() const;
    AmplitudeAxis amplitude_axis() const;
    void display_scale(Tfr::FreqAxis);
    void amplitude_axis(AmplitudeAxis);

    float targetSampleRate() const;
    void targetSampleRate(float);

    Tfr::TransformDesc::Ptr transform_desc() const;
    void transform_desc(Tfr::TransformDesc::Ptr);

    const TfrMapping& tfr_mapping() const;

    float length() const;
    void length(float L);

    int channels() const;
    void channels(int value);

    typedef VolatilePtr<Heightmap::Collection>::Ptr pCollection;
    typedef std::vector<pCollection> Collections;
    Collections collections() const;

private:
    void updateCollections();

    TfrMapping  tfr_mapping_;
    Collections collections_;
    float       length_;

public:
    static void test();
    static TfrMap::Ptr testInstance();
};

} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPING_H
