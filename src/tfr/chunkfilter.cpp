#include "chunkfilter.h"

namespace Tfr {

void ChunkFilterDesc::
        transformDesc(pTransformDesc d)
{
    transform_desc_ = d;
}
//virtual ChunkFilterDesc::Ptr    copy() const = 0;

pTransformDesc ChunkFilterDesc::
        transformDesc() const
{
    return transform_desc_;
}


} // namespace Tfr
