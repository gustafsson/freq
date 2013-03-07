#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {

Block::
        Block( const Reference& ref, const BlockConfiguration& block_config)
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
    to_delete( false ),
#endif
    ref_(ref),
    block_config_(block_config),
    block_interval_( ReferenceInfo(ref, block_config).getInterval() ),
    region_( ReferenceRegion(block_config)(ref) ),
    sample_rate_( ReferenceInfo(ref, block_config).sample_rate() )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
        TaskTimer tt(boost::format("Deleting block %s %s") % ref_ % ReferenceInfo(ref_, block_config_));
        glblock.reset();
    }
}


} // namespace Heightmap
