#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {


Block::
        Block( Reference ref, BlockLayout block_layout, VisualizationParams::ConstPtr visualization_params)
    :
    frame_number_last_used(-1),
    new_data_available( false ),
    block_data_(new BlockData),
    ref_(ref),
    block_layout_(block_layout),
    visualization_params_(visualization_params),
    block_interval_( ReferenceInfo(ref, block_layout_, visualization_params_).getInterval() ),
    region_( RegionFactory(block_layout_.block_size ())(ref) ),
    sample_rate_( ReferenceInfo(ref, block_layout_, visualization_params_).sample_rate() )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
        TaskTimer tt(boost::format("Deleting block %s %s") % ref_ % ReferenceInfo(ref_, block_layout_, visualization_params_));
        glblock.reset();
    }
}


} // namespace Heightmap
