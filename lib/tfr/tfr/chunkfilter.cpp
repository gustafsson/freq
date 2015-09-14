#include "chunkfilter.h"
#include "tfr/transform.h"

#include "demangle.h"

namespace Tfr {

Signal::OperationDesc::Extent ChunkFilterDesc::
        extent() const
{
    return Signal::OperationDesc::Extent();
}


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


ChunkFilterDesc::ptr ChunkFilterDesc::
        copy() const
{
    EXCEPTION_ASSERTX(false, vartype(*this) + "::copy not implemented");

    return ChunkFilterDesc::ptr();
}


QString ChunkFilterDesc::
        toString() const
{
    return (vartype(*this) + " on " + transform_desc_->toString ()).c_str();
}


} // namespace Tfr
