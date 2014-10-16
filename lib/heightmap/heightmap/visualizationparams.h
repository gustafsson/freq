#ifndef HEIGHTMAP_VISUALIZATIONPARAMS_H
#define HEIGHTMAP_VISUALIZATIONPARAMS_H

#include "shared_state.h"
#include "freqaxis.h"
#include "amplitudeaxis.h"

#include <memory>

namespace Heightmap {

class DetailInfo {
public:
    typedef std::shared_ptr<DetailInfo> ptr;

    virtual bool operator==(const DetailInfo&) const = 0;
    /**
     * @brief displayedTimeResolution describes the time resolution in ["time units" / "1 data point"].
     * @param FS
     * @param hz Cwt has different time resolutions depending on the frequency. Stft has a constant.
     * @return
     */
    virtual float displayedTimeResolution( float FS, float hz ) const = 0;

    /**
     * @brief displayedFrequencyResolution describes the frequency resolution in "data points" between hz1 and hz2.
     * @param FS
     * @param hz1
     * @param hz2
     * @return
     */
    virtual float displayedFrequencyResolution( float FS, float hz1, float hz2 ) const = 0;
};

/**
 * @brief The VisualizationParams class should describe all parameters that
 * define how waveform data turns into pixels on a heightmap.
 *
 * All methods are thread-safe without risking to wait for a long lock.
 * shared_state<details> is private to guarantee that the transient locks
 * created internally are the only locks on VisualizationParams.
 */
class VisualizationParams {
public:
    typedef std::shared_ptr<VisualizationParams> ptr;
    typedef std::shared_ptr<const VisualizationParams> const_ptr;

    VisualizationParams();

    bool operator==(const VisualizationParams& b) const;
    bool operator!=(const VisualizationParams& b) const;

    /**
     * Not that this is for the data that should be used. Blocks computed by
     * an old data source might still exist as they are being processed.
     */
    DetailInfo::ptr detail_info() const;
    void detail_info(DetailInfo::ptr);

    /**
     * Heightmap blocks are rather agnostic to FreqAxis. But it's needed to
     * create them.
     */
    FreqAxis display_scale() const;
    void display_scale(FreqAxis);

    /**
     * Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But
     * it's needed to create them.
     */
    AmplitudeAxis amplitude_axis() const;
    void amplitude_axis(AmplitudeAxis);

private:
    struct details {
        struct shared_state_traits: shared_state_traits_default {
            // Has only simple accessors, a simple mutex is faster than a more complex one
            typedef shared_state_mutex_notimeout_noshared shared_state_mutex;
        };

        FreqAxis display_scale_;
        AmplitudeAxis amplitude_axis_;
    };

    DetailInfo::ptr detail_info_;
    shared_state<details> details_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_VISUALIZATIONPARAMS_H
