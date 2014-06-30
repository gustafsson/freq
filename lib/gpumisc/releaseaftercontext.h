#ifndef RELEASEAFTERCONTEXT_H
#define RELEASEAFTERCONTEXT_H

#include <boost/noncopyable.hpp>

template<typename T>
class ReleaseAfterContext
{
public:
    ReleaseAfterContext(T& obj, void (T::*releaseCall)())
        :
        obj_(obj),
        releaseCall_(releaseCall)
    {}


    ~ReleaseAfterContext()
    {
        (obj_.*releaseCall_)();
    }


private:
    T& obj_;
    void (T::*releaseCall_)();
};

#endif // RELEASEAFTERCONTEXT_H
