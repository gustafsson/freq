#ifndef SIGNAL_PROCESSING_IINVALIDATOR_H
#define SIGNAL_PROCESSING_IINVALIDATOR_H

#include "signal/intervals.h"
#include "volatileptr.h"

namespace Signal {
namespace Processing {

/**
 * @brief The IInvalidator interface should invalidate step cache and its implementation specific dependencies.
 *
 * It should be accessed from multiple threads. So use VolatilePtr.
 */
class IInvalidator
{
public:
    typedef VolatilePtr<IInvalidator> Ptr;
    typedef Ptr::WritePtr WritePtr;
    typedef Ptr::ReadPtr ReadPtr;

    virtual ~IInvalidator() {}

    virtual void deprecateCache(Signal::Intervals what) const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_IINVALIDATOR_H
