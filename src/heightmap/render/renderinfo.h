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

class RenderInfo
{
public:
    enum LevelOfDetal {
        Lod_NeedBetterF,
        Lod_NeedBetterT,
        Lod_Ok,
        Lod_Invalid
    };

    RenderInfo(glProjection* gl_projection, BlockLayout bl, VisualizationParams::ConstPtr vp, FrustumClip* frustum_clip, float redundancy);

    RenderInfo::LevelOfDetal testLod( Reference ref ) const;
    bool boundsCheck( Reference ref, ReferenceInfo::BoundsCheck) const;

    Reference findRefAtCurrentZoomLevel( Heightmap::Position p, Reference entireHeightmap ) const;
private:
    glProjection* gl_projection;
    BlockLayout bl;
    VisualizationParams::ConstPtr vp;
    FrustumClip* frustum_clip;
    float redundancy;

    bool computePixelsPerUnit( Region r, float& timePixels, float& scalePixels ) const;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERINFO_H
