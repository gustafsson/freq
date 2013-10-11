#include "renderheightmap.h"
#include "TaskTimer.h"
#include "renderinfo.h"

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

namespace Heightmap {
namespace Render {

RenderHeightmap::
        RenderHeightmap(BlockCache::Ptr cache, glProjection* gl_projection, RenderBlock* render_block)
    :
      cache_(cache),
      gl_projection_(gl_projection),
      render_block_(render_block)
{
}


void RenderHeightmap::
        render( Reference ref, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy )
{
    if (!renderChildrenSpectrogramRef( ref, bl, vp, frustum_clip, redundancy ))
        renderSpectrogramRef( ref );
}


bool RenderHeightmap::
        renderChildrenSpectrogramRef( Reference ref, BlockLayout bl, VisualizationParams::ConstPtr vp, const FrustumClip& frustum_clip, float redundancy )
{
    TIME_RENDERER_BLOCKS TaskTimer tt(boost::format("%s")
          % ReferenceInfo(ref, bl, vp));

    RenderInfo::LevelOfDetal lod = RenderInfo(gl_projection_).testLod (ref, bl, vp, frustum_clip, redundancy);
    switch(lod) {
    case RenderInfo::Lod_NeedBetterF:
        renderChildrenSpectrogramRef( ref.bottom(), bl, vp, frustum_clip, redundancy );
        renderChildrenSpectrogramRef( ref.top(), bl, vp, frustum_clip, redundancy );
        break;
    case RenderInfo::Lod_NeedBetterT:
        renderChildrenSpectrogramRef( ref.left(), bl, vp, frustum_clip, redundancy );
        if (ReferenceInfo(ref.right (), bl, vp)
                .boundsCheck(ReferenceInfo::BoundsCheck_OutT))
            renderChildrenSpectrogramRef( ref.right(), bl, vp, frustum_clip, redundancy );
        break;
    case RenderInfo::Lod_Ok:
        renderSpectrogramRef( ref );
        break;
    case RenderInfo::Lod_Invalid: // ref is not within the current view frustum
        return false;
    }

    return true;
}


void RenderHeightmap::
        renderSpectrogramRef( Reference ref )
{
    try {
        BlockCache::WritePtr cache( cache_, 0 );
        pBlock block = cache->find( ref );

        if (block) {
            if (!render_block_->renderBlock(block)) {
                render_block_->renderBlockError(block->block_layout (), block->getRegion ());
            }
        }
    } catch (const BlockCache::LockFailed&) {
        // ok, skip
    }
}


void RenderHeightmap::
        test()
{

}


} // namespace Render
} // namespace Heightmap
