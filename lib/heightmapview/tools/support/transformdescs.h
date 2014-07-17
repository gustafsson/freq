#ifndef TOOLS_SUPPORT_TRANSFORMDESCS_H
#define TOOLS_SUPPORT_TRANSFORMDESCS_H

#include "shared_state.h"
#include "tfr/transform.h"

#include <set>

namespace Tools {
namespace Support {

class TransformDescs
{
public:
    typedef shared_state<TransformDescs> ptr;

    /**
     * @brief getParam always returns an instance.
     * @return an instance of type T.
     */
    template<typename T>
    T& getParam() {
        return *dynamic_cast<T*>(getParamPtr<T>().get());
    }

    /**
     * @brief cloneType returns a copy of an existing parameters of a
     * certain type. May return null.
     * @param i the type to look for.
     * @return a unique TransformDesc::Ptr or a null pointer if the
     * requested type was not found.
     */
    Tfr::TransformDesc::ptr cloneType(const std::type_info& i) const;

private:
    template<typename T>
    Tfr::TransformDesc::ptr getParamPtr() {
        for (Tfr::TransformDesc::ptr p : descriptions_)
            if (dynamic_cast<volatile T*>(p.get()))
                return p;

        Tfr::TransformDesc::ptr p(new T());
        descriptions_.insert(p);
        return p;
    }

    std::set<Tfr::TransformDesc::ptr> descriptions_;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_TRANSFORMDESCS_H
