#include "block.h"
#include "glblock.h"

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
    region_( RegionFactory(block_layout_)(ref) ),
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


bool Block::
        update_glblock_data()
{
    bool r = false;

    try {
        // Lock if available but don't wait for it to become available
        BlockData::ReadPtr bd(block_data (), 0);

        if (new_data_available) {
            *glblock->height()->data = *bd->cpu_copy; // 256 KB memcpy < 100 us (256*256*4 = 256 KB, about 52 us)
            new_data_available = false;

            r = true;
        }
    } catch (const BlockData::LockFailed&) {}

    return r;
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
