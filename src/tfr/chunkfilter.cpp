#include "chunkfilter.h"

#include "demangle.h"

namespace Tfr {

void ChunkFilterDesc::
        transformDesc(pTransformDesc d)
{
    transform_desc_ = d;
}


pTransformDesc ChunkFilterDesc::
        transformDesc() const
{
    return transform_desc_;
}


ChunkFilterDesc::Ptr ChunkFilterDesc::
        copy() const
{
    EXCEPTION_ASSERTX(false, vartype(*this) + "::copy not implemented");

    return ChunkFilterDesc::Ptr();
}


} // namespace Tfr
