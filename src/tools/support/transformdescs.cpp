#include "transformdescs.h"

namespace Tools {
namespace Support {


Tfr::TransformDesc::Ptr TransformDescs::
        cloneType(const std::type_info& i) const
{
    foreach (Tfr::TransformDesc::Ptr p, descriptions_)
        if (typeid(*p.get()) == i)
            return p->copy ();
    return Tfr::TransformDesc::Ptr();
}


} // namespace Support
} // namespace Tools
