#include "transformdescs.h"

namespace Tools {
namespace Support {


Tfr::TransformDesc::ptr TransformDescs::
        cloneType(const std::type_info& i) const
{
    for (Tfr::TransformDesc::ptr p : descriptions_)
    {
        auto pv = p.get ();
        if (typeid(*pv) == i)
            return p->copy ();
    }
    return Tfr::TransformDesc::ptr();
}


} // namespace Support
} // namespace Tools
