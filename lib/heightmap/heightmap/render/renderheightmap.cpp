#include "renderheightmap.h"
#include "tasktimer.h"
#include "renderinfo.h"
#include "heightmap/reference_hash.h"
#include "unused.h"

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

namespace Heightmap {
namespace Render {


RenderHeightmap::
        RenderHeightmap(RenderInfo* render_info)
    :
      render_info(render_info)
{
    // TODO define usage
    UNUSED(RenderInfo*r) = this->render_info;
}




void RenderHeightmap::
        test()
{

}


} // namespace Render
} // namespace Heightmap
