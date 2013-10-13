#include "renderset.h"
#include "heightmap/reference_hash.h"


namespace Heightmap {
namespace Render {


RenderSet::references_t& operator|=(RenderSet::references_t& A, const RenderSet::references_t& B) {
    A.insert(B.begin(), B.end());
    return A;
}


RenderSet::references_t& operator|=(RenderSet::references_t& A, const Heightmap::Reference& r) {
    A.insert(r);
    return A;
}


RenderSet::
        RenderSet(RenderInfoI* render_info)
    :
      render_info(render_info)
{
}


RenderSet::references_t RenderSet::
        computeRenderSet(Reference entireHeightmap)
{
    references_t R = computeChildrenRenderSet( entireHeightmap );
    if (R.empty()) {
        R |= entireHeightmap;
    }
    return R;
}


Reference RenderSet::
        computeRefAt( Heightmap::Position p, Reference entireHeightmap ) const
{
    Reference ref = entireHeightmap;

    // The first 'ref' will be a super-ref containing all other refs, thus
    // containing 'p' too. This while-loop zooms in on a ref containing
    // 'p' with enough details.

    // If 'p' is not within entireHeightmap this algorithm will choose some ref
    // along the border closest to the point 'p'.

    while (true)
    {
        RenderInfoI::LevelOfDetal lod = render_info->testLod(ref);

        Region r = render_info->region(ref);

        switch(lod)
        {
        case RenderInfoI::Lod_NeedBetterF:
            if ((r.a.scale+r.b.scale)/2 > p.scale)
                ref = ref.bottom();
            else
                ref = ref.top();
            break;

        case RenderInfoI::Lod_NeedBetterT:
            if ((r.a.time+r.b.time)/2 > p.time)
                ref = ref.left();
            else
                ref = ref.right();
            break;

        default:
            return ref;
        }
    }
}


RenderSet::references_t RenderSet::
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
        //if (render_info->boundsCheck(ref.right (), ReferenceInfo::BoundsCheck_OutT))
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


void RenderSet::
        test()
{
    {
        // The RenderSet class should compute which references that cover the
        // visible heightmap with sufficient resolution, as judged by RenderInfoI.
    }
}

} // namespace Render
} // namespace Heightmap
