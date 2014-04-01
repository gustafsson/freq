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

class RenderInfoI
{
public:
    enum LevelOfDetal {
        Lod_NeedBetterF,
        Lod_NeedBetterT,
        Lod_Ok,
        Lod_Invalid
    };

    virtual ~RenderInfoI() {}

    virtual RenderInfoI::LevelOfDetal testLod( Reference ref ) const = 0;
    virtual Region                    region(Reference ref) const = 0;
};


class RenderInfo: public RenderInfoI
{
public:
    RenderInfo(glProjection* gl_projection, BlockLayout bl, VisualizationParams::const_ptr vp, FrustumClip* frustum_clip, float redundancy);

    RenderInfoI::LevelOfDetal   testLod( Reference ref ) const;
    Region                      region(Reference ref) const;

private:
    glProjection* gl_projection;
    BlockLayout bl;
    VisualizationParams::const_ptr vp;
    FrustumClip* frustum_clip;
    float redundancy;

    bool boundsCheck( Reference ref, ReferenceInfo::BoundsCheck) const;
    bool computePixelsPerUnit( Region r, float& timePixels, float& scalePixels ) const;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERINFO_H
