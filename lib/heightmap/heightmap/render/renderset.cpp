#include "renderset.h"
#include "heightmap/reference_hash.h"
#include "tasktimer.h"

//#define LOG_TRAVERSAL
#define LOG_TRAVERSAL if(0)

namespace Heightmap {
namespace Render {


RenderSet::references_t& operator|=(RenderSet::references_t& A, const RenderSet::references_t& B) {
    A.insert(B.begin(), B.end());
    return A;
}


RenderSet::references_t& operator|=(RenderSet::references_t& A, const RenderSet::references_t::value_type& r) {
    A.insert(r);
    return A;
}


RenderSet::references_t RenderSet::
        makeSet(Reference r, LevelOfDetail lod)
{
    return RenderSet::references_t{RenderSet::references_t::value_type(r,lod)};
}


RenderSet::
        RenderSet(RenderInfo* render_info, float L)
    :
      render_info(render_info),
      L(L)
{
}


RenderSet::references_t RenderSet::
        computeRenderSet(Reference entireHeightmap)
{
    if (0.f == L)
        return makeSet (entireHeightmap);

    references_t R = computeChildrenRenderSet( entireHeightmap );
    if (R.empty()) {
        R |= makeSet (entireHeightmap);
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
        LevelOfDetail lod = render_info->testLod(ref);

        Region r = render_info->region(ref);

        if (lod.need_s ())
        {
            if ((r.a.scale+r.b.scale)/2 > p.scale)
                ref = ref.bottom();
            else
                ref = ref.top();
        }
        else if (lod.need_t ())
        {
            if ((r.a.time+r.b.time)/2 > p.time)
                ref = ref.left();
            else
                ref = ref.right();
        }
        else
        {
            return ref;
        }
    }
}


RenderSet::references_t RenderSet::
        computeChildrenRenderSet( Reference ref )
{
    references_t R;

    LevelOfDetail lod = render_info->testLod (ref );
    if (lod.need_s ())
    {
        LOG_TRAVERSAL TaskInfo ti(boost::format("renderset %s need_s, s=%g, t=%g")
                    % render_info->region (ref) % lod.s () % lod.t ());

        R |= computeChildrenRenderSet( ref.bottom() );
        R |= computeChildrenRenderSet( ref.top() );
    } else if (lod.need_t ()) {
        LOG_TRAVERSAL TaskInfo ti(boost::format("renderset %s need_t, s=%s, t=%g")
                    % render_info->region (ref) % lod.s () % lod.t ());

        R |= computeChildrenRenderSet( ref.left() );
        if ( render_info->region(ref.right (),false).a.time < L) {
            R |= computeChildrenRenderSet( ref.right() );
        }
    } else if (lod.ok ()) {
        R |= references_t::value_type(ref,lod);
    } else {
        LOG_TRAVERSAL TaskInfo ti(boost::format("renderset %s invalid, s=%s, t=%g")
                    % render_info->region (ref) % lod.s () % lod.t ());
        // ref is not within the current view frustum
    }

    return R;
}


void RenderSet::
        test()
{
    {
        // The RenderSet class should compute which references that cover the
        // visible heightmap with sufficient resolution, as judged by RenderInfo.
    }
}

} // namespace Render
} // namespace Heightmap
