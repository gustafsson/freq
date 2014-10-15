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
        release();
    }


    void release() {
        if (releaseCall_) {
            (obj_.*releaseCall_)();
            releaseCall_ = 0;
        }
    }

private:
    T& obj_;
    void (T::*releaseCall_)();
};

#endif // RELEASEAFTERCONTEXT_H
