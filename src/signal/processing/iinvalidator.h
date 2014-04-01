#ifndef SIGNAL_PROCESSING_IINVALIDATOR_H
#define SIGNAL_PROCESSING_IINVALIDATOR_H

#include "signal/intervals.h"
#include "shared_state.h"

namespace Signal {
namespace Processing {

/**
 * @brief The IInvalidator interface should invalidate step cache and its implementation specific dependencies.
 *
 * It should be accessed from multiple threads. So use shared_state.
 */
class IInvalidator
{
public:
    typedef shared_state<IInvalidator> Ptr;

    virtual ~IInvalidator() {}

    virtual void deprecateCache(Signal::Intervals what) const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_IINVALIDATOR_H
