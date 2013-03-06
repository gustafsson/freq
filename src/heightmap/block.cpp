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
    region_( ref.getRegion() ),
    sample_rate_( ref.sample_rate() )
{
}


Block::
        Block( Signal::Interval block_interval, Region region, float sample_rate )
    :
    frame_number_last_used(-1),
#ifndef SAWE_NO_MUTEX
    new_data_available( false ),
    to_delete( false ),
#endif
    ref_( BlockConfiguration( BlockSize(2,2), 1) ),
    block_interval_( block_interval ),
    region_( region ),
    sample_rate_( sample_rate )
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
