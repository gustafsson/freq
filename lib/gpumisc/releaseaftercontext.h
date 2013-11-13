#ifndef RELEASEAFTERCONTEXT_H
#define RELEASEAFTERCONTEXT_H

#include <boost/noncopyable.hpp>

template<typename T>
class ReleaseAfterContext
{
public:
    ReleaseAfterContext(const T& obj, void (T::*releaseCall)() const)
        :
        obj_(obj),
        releaseCall_(releaseCall)
    {}


    ~ReleaseAfterContext()
    {
        (obj_.*releaseCall_)();
    }


private:
    const T& obj_;
    void (T::*releaseCall_)() const;
};

#endif // RELEASEAFTERCONTEXT_H
