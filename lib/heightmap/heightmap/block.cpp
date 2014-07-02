#include "block.h"
#include "GlTexture.h"

#include "tasktimer.h"
#include "log.h"


//#define BLOCK_INFO
#define BLOCK_INFO if(0)

namespace Heightmap {


Block::
        Block( Reference ref, BlockLayout block_layout, VisualizationParams::const_ptr visualization_params, std::shared_ptr<GlTexture> texture)
    :
    frame_number_last_used(0),
    ref_(ref),
    block_layout_(block_layout),
    block_interval_( ReferenceInfo(ref, block_layout, visualization_params).getInterval() ),
    region_( RegionFactory(block_layout)(ref) ),
    sample_rate_( ReferenceInfo(ref, block_layout, visualization_params).sample_rate() ),
    visualization_params_(visualization_params),
    texture_(texture)
{
    if (texture_)
    {
        EXCEPTION_ASSERT_EQUALS(texture_->getWidth (), block_layout.texels_per_row ());
        EXCEPTION_ASSERT_EQUALS(texture_->getHeight (), block_layout.texels_per_column ());
    }
}




void Block::
        test()
{
    // It should store information and data about a block.
    {
        // ...
    }
}

} // namespace Heightmap
