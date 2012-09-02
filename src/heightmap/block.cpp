#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {

Block::
        Block( Reference ref )
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
#endif
    ref_(ref),
    block_interval_( ref.getInterval() ),
    region_( ref.getRegion() ),
    sample_rate_( ref.sample_rate() )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
        TaskTimer tt("Deleting block %s", ref_.toString().c_str());
        glblock.reset();
    }
}


} // namespace Heightmap
