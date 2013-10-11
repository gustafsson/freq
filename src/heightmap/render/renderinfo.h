#ifndef HEIGHTMAP_RENDER_RENDERINFO_H
#define HEIGHTMAP_RENDER_RENDERINFO_H

#include "glprojection.h"
#include "heightmap/reference.h"
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

    RenderInfo(glProjection* gl_projection);

    RenderInfo::LevelOfDetal testLod( Reference ref, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy );
    Reference findRefAtCurrentZoomLevel( Heightmap::Position p, Reference entireHeightmap, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy );
private:
    glProjection* gl_projection;

    bool computePixelsPerUnit( Region r, const FrustumClip& frustum_clip, float& timePixels, float& scalePixels );
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERINFO_H
