#include "block.h"
#include "glblock.h"

#include "TaskTimer.h"


//#define BLOCK_INFO
#define BLOCK_INFO if(0)

namespace Heightmap {


Block::
        Block( Reference ref, BlockLayout block_layout, VisualizationParams::ConstPtr visualization_params)
    :
    frame_number_last_used(-1),
    block_data_(new BlockData),
    new_data_available_(false),
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


BlockData::WritePtr Block::
        block_data()
{
    BlockData::WritePtr b(block_data_);
    new_data_available_ = true;
    return b;
}


void Block::
        discard_new_data_available()
{
    BlockData::WritePtr b(block_data_);
    new_data_available_ = false;
}


bool Block::
        update_glblock_data()
{
    // Lock if available but don't wait for it to become available
    BlockData::WritePtr bd(block_data_, NoLockFailed());

    if (bd.get () && new_data_available_) {
        BLOCK_INFO TaskTimer tt(boost::format("Updating glblock height %s %s") % block_interval_ % region_);

        *glblock->height()->data = *bd->cpu_copy; // 256 KB memcpy < 100 us (256*256*4 = 256 KB, about 52 us)
        new_data_available_ = false;

        return true;
    }

    return false;
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
