#include "transformdescs.h"

namespace Tools {
namespace Support {


Tfr::TransformDesc::ptr TransformDescs::
        cloneType(const std::type_info& i) const
{
    for (Tfr::TransformDesc::ptr p : descriptions_)
        if (typeid(*p.get()) == i)
            return p->copy ();
    return Tfr::TransformDesc::ptr();
}


} // namespace Support
} // namespace Tools
