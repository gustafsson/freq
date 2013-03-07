#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {

Block::
        Block( const Reference& ref, const TfrMapping& tfr_mapping)
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
    to_delete( false ),
#endif
    ref_(ref),
    tfr_mapping_(tfr_mapping),
    block_interval_( ReferenceInfo(ref, tfr_mapping).getInterval() ),
    region_( ReferenceRegion(tfr_mapping)(ref) ),
    sample_rate_( ReferenceInfo(ref, tfr_mapping).sample_rate() )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
        TaskTimer tt(boost::format("Deleting block %s %s") % ref_ % ReferenceInfo(ref_, tfr_mapping_));
        glblock.reset();
    }
}


} // namespace Heightmap
