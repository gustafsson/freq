#ifndef HEIGHTMAP_VISUALIZATIONPARAMS_H
#define HEIGHTMAP_VISUALIZATIONPARAMS_H

#include "shared_state.h"
#include "tfr/transform.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"

#include <memory>

namespace Heightmap {


/**
 * @brief The VisualizationParams class should describe all parameters that
 * define how waveform data turns into pixels on a heightmap.
 *
 * All methods are thread-safe without risking to wait for a long lock.
 * VolatilePtr is private to guarantee that the transient locks created
 * internally are the only locks on VisualizationParams.
 */
class VisualizationParams {
public:
    typedef std::shared_ptr<VisualizationParams> ptr;
    typedef std::shared_ptr<const VisualizationParams> const_ptr;

    VisualizationParams();

    bool operator==(const VisualizationParams& b) const;
    bool operator!=(const VisualizationParams& b) const;

    /**
     * Not that this is the transform that should be used. Blocks computed by
     * an old transform desc might still exist as they are being processed.
     */
    Tfr::TransformDesc::ptr transform_desc() const;
    void transform_desc(Tfr::TransformDesc::ptr);

    /**
     * Heightmap blocks are rather agnostic to FreqAxis. But it's needed to
     * create them.
     */
    Tfr::FreqAxis display_scale() const;
    void display_scale(Tfr::FreqAxis);

    /**
     * Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But
     * it's needed to create them.
     */
    AmplitudeAxis amplitude_axis() const;
    void amplitude_axis(AmplitudeAxis);

private:
    struct details {
        Tfr::TransformDesc::ptr transform_desc_;
        Tfr::FreqAxis display_scale_;
        AmplitudeAxis amplitude_axis_;
    };

    shared_state<details> details_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_VISUALIZATIONPARAMS_H
