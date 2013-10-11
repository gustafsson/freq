#ifndef HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H
#define HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H

#include "heightmap/blockcache.h"
#include "heightmap/render/frustumclip.h"
#include "glprojection.h"
#include "renderblock.h"

namespace Heightmap {
namespace Render {

/**
 * @brief The RenderHeightmap class should render a block cache within a frustum.
 */
class RenderHeightmap
{
public:
    RenderHeightmap(BlockCache::Ptr cache, glProjection* gl_projection, RenderBlock* render_block);

    void render( Reference ref, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy );

private:
    BlockCache::Ptr cache_;
    glProjection* gl_projection_;
    RenderBlock* render_block_;

    void renderSpectrogramRef( Reference ref );
    bool renderChildrenSpectrogramRef( Reference ref, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy );

public:
    static void test();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H
