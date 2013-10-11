#include "renderheightmap.h"
#include "TaskTimer.h"
#include "renderinfo.h"
#include "heightmap/reference_hash.h"

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

namespace Heightmap {
namespace Render {


RenderHeightmap::references_t& operator|=(RenderHeightmap::references_t& A, const RenderHeightmap::references_t& B) {
    A.insert(B.begin(), B.end());
    return A;
}


RenderHeightmap::references_t& operator|=(RenderHeightmap::references_t& A, const Heightmap::Reference& r) {
    A.insert(r);
    return A;
}


RenderHeightmap::
        RenderHeightmap(RenderInfo* render_info)
    :
      render_info(render_info)
{
}


RenderHeightmap::references_t RenderHeightmap::
        computeRenderSet( Reference ref )
{
    references_t R = computeChildrenRenderSet( ref );
    if (R.empty()) {
        R |= ref;
    }
    return R;
}


RenderHeightmap::references_t RenderHeightmap::
        computeChildrenRenderSet( Reference ref )
{
    references_t R;

    RenderInfo::LevelOfDetal lod = render_info->testLod (ref );
    switch(lod) {
    case RenderInfo::Lod_NeedBetterF:
        R |= computeChildrenRenderSet( ref.bottom() );
        R |= computeChildrenRenderSet( ref.top() );
        break;
    case RenderInfo::Lod_NeedBetterT:
        R |= computeChildrenRenderSet( ref.left() );
        if (render_info->boundsCheck(ref.right (), ReferenceInfo::BoundsCheck_OutT))
            R |= computeChildrenRenderSet( ref.right() );
        break;
    case RenderInfo::Lod_Ok:
        R |= ref;
        break;
    case RenderInfo::Lod_Invalid: // ref is not within the current view frustum
        break;
    }

    return R;
}


void RenderHeightmap::
        test()
{

}


} // namespace Render
} // namespace Heightmap
