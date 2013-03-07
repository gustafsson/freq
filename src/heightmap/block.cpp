#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {

Block::
        Block( ReferenceInfo ref )
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
    to_delete( false ),
#endif
    ref_(ref.reference()),
    block_interval_( ref.getInterval() ),
    region_( ReferenceRegion(ref.block_config ())(ref.reference ()) ),
    sample_rate_( ref.sample_rate() ),
    block_size_( ref.block_config ().block_size () )
{
}


Block::
        Block( Signal::Interval block_interval, Region region, float sample_rate, BlockSize block_size )
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
    to_delete( false ),
#endif
    ref_( BlockConfiguration( BlockSize(2,2), 1) ),
    block_interval_( block_interval ),
    region_( region ),
    sample_rate_( sample_rate ),
    block_size_( block_size )
{
}


Block::
        ~Block()
{
    if (glblock)
    {
        TaskTimer tt(boost::format("Deleting block %s %s") % ref_ % ReferenceRegion(block_size_)(ref_));
        glblock.reset();
    }
}


} // namespace Heightmap
