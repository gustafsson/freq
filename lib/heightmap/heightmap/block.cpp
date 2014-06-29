#include "block.h"
#include "render/glblock.h"

#include "tasktimer.h"
#include "log.h"


//#define BLOCK_INFO
#define BLOCK_INFO if(0)

namespace Heightmap {


Block::
        Block( Reference ref, BlockLayout block_layout, VisualizationParams::const_ptr visualization_params)
    :
    frame_number_last_used(0),
    ref_(ref),
    block_layout_(block_layout),
    visualization_params_(visualization_params),
    block_interval_( ReferenceInfo(ref, block_layout_, visualization_params_).getInterval() ),
    region_( RegionFactory(block_layout_)(ref) ),
    sample_rate_( ReferenceInfo(ref, block_layout_, visualization_params_).sample_rate() )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
//        TaskTimer tt(boost::format("Deleting block %s %s") % ref_ % ReferenceInfo(ref_, block_layout_, visualization_params_));
        glblock.reset();
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
