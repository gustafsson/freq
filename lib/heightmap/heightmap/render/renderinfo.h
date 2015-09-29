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
    LevelOfDetail(bool valid)
        : LevelOfDetail(valid?1:-1, valid?1:-1, true, true)
    {}

    LevelOfDetail(double t, double s,
                  bool max_t, bool max_s)
        :
          t(t), s(s),
          max_t(max_t), max_s(max_s)
    {}

    /**
     * @brief need_t calculates if a higher time resolution is needed.
     * if a higher time resolution is needed but a higher scale resolution is
     * even more needed need_t returns false.
     * @return true if a higher resolution is needed
     */
    bool need_t() const { return (t >= s || max_s) && t > 1 && !max_t; }
    bool need_s() const { return (s >= t || max_t) && s > 1 && !max_s; }

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
    bool valid() const { return t>=0 && s>=0; }

    /**
     * @brief t how many pixels that spans one step along the t axis in the underlying block data (the texels).
     * Note that the value is relative to the underlying block data (texels), not to the originally calculated transform.
     *
     * t is the resolution in the most detailed corner of a ref
     * @return pixels/texel
     */
    const double t;
    const double s;

private:
    const bool max_t;
    const bool max_s;
};

class CornerResolution final {
public:
    // build from texels per pixel
    CornerResolution(float x00, float x01, float x10, float x11, float y00, float y01, float y10, float y11);

    // mipmap level for each corner = log2(texels/pixel). >= 0.
    const float x00; // texels/pixel, >= 0.5
    const float x01;
    const float x10;
    const float x11;
    const float y00;
    const float y01;
    const float y10;
    const float y11;
};

class RenderInfo
{
public:
    RenderInfo(const glProjecter* gl_projecter, BlockLayout bl, VisualizationParams::const_ptr vp, float redundancy);

    LevelOfDetail       testLod (Reference ref) const;
    CornerResolution    cornerResolution (Reference ref) const;
    Region              visible_region (Reference ref) const;

private:
    const glProjecter* gl_projecter;
    Render::FrustumClip frustum_clip;
    BlockLayout bl;
    VisualizationParams::const_ptr vp;
    float redundancy;

    bool boundsCheck( Reference ref, ReferenceInfo::BoundsCheck) const;
    bool computePixelsPerUnit( Region r, double& timePixels, double& scalePixels ) const;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERINFO_H
