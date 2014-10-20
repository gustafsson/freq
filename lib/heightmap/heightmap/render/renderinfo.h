#ifndef HEIGHTMAP_RENDER_RENDERINFO_H
#define HEIGHTMAP_RENDER_RENDERINFO_H

#include "glprojection.h"
#include "heightmap/reference.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/position.h"
#include "heightmap/blocklayout.h"
#include "heightmap/visualizationparams.h"
#include "heightmap/render/frustumclip.h"

namespace Heightmap {
namespace Render {

/**
 * @brief The LevelOfDetail class describes information about the level of
 * detailed for a Reference as computed by RenderInfo.
 *
 * ok() && s()<1 && t()<1: screen pixels are larger than block data points along both the S axis and the T axis
 * ok() && s()>1 && t()<1: screen pixels are smaller than block data points along S but not T
 * ok() && s()<1 && t()>1: screen pixels are smaller than block data points along T but not S
 * ok() && s()>1 && t()>1: screen pixels are smaller than block data points along both T and S
*/
class LevelOfDetail {
public:
    LevelOfDetail(bool valid);
    LevelOfDetail(double pixels_per_data_points_t, double pixels_per_data_points_s,
                  bool max_t, bool max_s);
    /**
     * @brief need_t calculates if a higher time resolution is needed.
     * if a higher time resolution is needed but a higher scale resolution is
     * even more needed need_t returns false.
     * @return true if a higher resolution is needed
     */
    bool need_t() const { return (t() >= s() || max_s) && t() > 1 && !max_t; }
    bool need_s() const { return (s() >= t() || max_t) && s() > 1 && !max_s; }

    /**
     * @brief ok calculates if this reference has a high enough time and scale
     * resolution
     * @return true if high enough
     */
    bool ok() const { return !need_t() && !need_s() && valid(); }

    /**
     * @brief valid describes whether this instance contains valid values.
     * The default constructor produces an invalid instance.
     * @return true if valid
     */
    bool valid() const;

    /**
     * @brief t how many pixels that spans one step along the t axis in the underlying block data.
     * Note that the value is relative to the underlying block data, not to the originally calculated transform.
     * @return pixels/datapoint
     */
    double t() const { return pixels_per_data_point_t; }
    double s() const { return pixels_per_data_point_s; }

private:
    const double pixels_per_data_point_t;
    const double pixels_per_data_point_s;
    const bool max_t;
    const bool max_s;
};


class RenderInfo
{
public:
    RenderInfo(const glProjection* gl_projection, BlockLayout bl, VisualizationParams::const_ptr vp, float redundancy);

    LevelOfDetail       testLod (Reference ref) const;
    Region              region (Reference ref, bool render_region=true) const;

private:
    const glProjection* gl_projection;
    BlockLayout bl;
    VisualizationParams::const_ptr vp;
    float redundancy;

    bool boundsCheck( Reference ref, ReferenceInfo::BoundsCheck) const;
    bool computePixelsPerUnit( Region r, float& timePixels, float& scalePixels ) const;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERINFO_H
