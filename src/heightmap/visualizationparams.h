#ifndef HEIGHTMAP_VISUALIZATIONPARAMS_H
#define HEIGHTMAP_VISUALIZATIONPARAMS_H

#include "volatileptr.h"
#include "tfr/transform.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"

namespace Heightmap {


/**
 * @brief The VisualizationParams class should describe all parameters that
 * define how waveform data turns into pixels on a heightmap.
 *
 * All methods are thread-safe without risking to wait for a long lock.
 * VolatilePtr is private to guarantee that the transient locks created
 * internally are the only locks on VisualizationParams.
 */
class VisualizationParams: private VolatilePtr<VisualizationParams> {
public:
    typedef boost::shared_ptr<volatile VisualizationParams> Ptr;
    typedef boost::shared_ptr<const volatile VisualizationParams> ConstPtr;

    VisualizationParams();

    bool operator==(const volatile VisualizationParams& b) const volatile;
    bool operator!=(const volatile VisualizationParams& b) const volatile;

    /**
     * Not that this is the transform that should be used. Blocks computed by
     * an old transform desc might still exist as they are being processed.
     */
    Tfr::TransformDesc::Ptr transform_desc() const volatile;
    void transform_desc(Tfr::TransformDesc::Ptr) volatile;

    /**
     * Heightmap blocks are rather agnostic to FreqAxis. But it's needed to
     * create them.
     */
    Tfr::FreqAxis display_scale() const volatile;
    void display_scale(Tfr::FreqAxis) volatile;

    /**
     * Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But
     * it's needed to create them.
     */
    AmplitudeAxis amplitude_axis() const volatile;
    void amplitude_axis(AmplitudeAxis) volatile;

private:
    friend class VolatilePtr<VisualizationParams>;

    Tfr::TransformDesc::Ptr transform_desc_;
    Tfr::FreqAxis display_scale_;
    AmplitudeAxis amplitude_axis_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_VISUALIZATIONPARAMS_H
