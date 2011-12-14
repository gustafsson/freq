#include "block.h"

#include "TaskTimer.h"

namespace Heightmap {

Block::
        Block( Reference ref )
    :
    frame_number_last_used(-1),
    ref(ref)
#ifndef SAWE_NO_MUTEX
    ,new_data_available( false )
#endif
{}


Block::
        ~Block()
{
    TaskInfo("Deleting block %s", ref.toString().c_str());
}


} // namespace Heightmap
