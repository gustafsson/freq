#ifndef SIGNAL_PROCESSING_IINVALIDATOR_H
#define SIGNAL_PROCESSING_IINVALIDATOR_H

#include "signal/intervals.h"
#include <memory>

namespace Signal {
namespace Processing {

/**
 * @brief The IInvalidator interface should invalidate step cache and its implementation specific dependencies.
 *
 * It should be accessed from multiple threads and thus it should be data-race free.
 */
class IInvalidator
{
public:
    typedef std::shared_ptr<IInvalidator> ptr;

    virtual ~IInvalidator() {}

    virtual void deprecateCache(Signal::Intervals what) const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_IINVALIDATOR_H
