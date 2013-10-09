#include "renderinfo.h"
#include "heightmap/reference.h"
#include "heightmap/position.h"

namespace Heightmap {
namespace Render {

RenderInfo::
        RenderInfo(glProjection* /*gl_projection*/)
{
}

/*
Reference RenderInfo::
        findRefAtCurrentZoomLevel( Heightmap::Position p )
{
    //Position max_ss = collection->max_sample_size();
    Reference ref = read1(collection)->entireHeightmap();
    BlockLayout bc = read1(collection)->block_layout ();

    // The first 'ref' will be a super-ref containing all other refs, thus
    // containing 'p' too. This while-loop zooms in on a ref containing
    // 'p' with enough details.

    // 'p' is assumed to be valid to start with. Ff they're not valid
    // this algorithm will choose some ref along the border closest to the
    // point 'p'.

    while (true)
    {
        LevelOfDetal lod = testLod(ref);

        Region r = RegionFactory(bc)(ref);

        switch(lod)
        {
        case Lod_NeedBetterF:
            if ((r.a.scale+r.b.scale)/2 > p.scale)
                ref = ref.bottom();
            else
                ref = ref.top();
            break;

        case Lod_NeedBetterT:
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
*/

} // namespace Render
} // namespace Heightmap
