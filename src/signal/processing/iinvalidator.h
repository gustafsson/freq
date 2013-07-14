#ifndef SIGNAL_PROCESSING_IINVALIDATOR_H
#define SIGNAL_PROCESSING_IINVALIDATOR_H

#include "signal/intervals.h"
#include "volatileptr.h"

namespace Signal {
namespace Processing {

class Step;

/**
 * @brief The IInvalidator interface should invalidate step cache and its implementation specific dependencies.
 *
 * It should be accessed from multiple threads. So use VolatilePtr.
 */
class IInvalidator: public VolatilePtr<IInvalidator>
{
public:
    virtual ~IInvalidator() {}

    virtual void deprecateCache(boost::shared_ptr<volatile Step> at, Signal::Intervals what) const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_IINVALIDATOR_H
