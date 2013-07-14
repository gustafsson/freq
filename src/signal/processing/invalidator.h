#ifndef SIGNAL_PROCESSING_INVALIDATOR_H
#define SIGNAL_PROCESSING_INVALIDATOR_H

#include "step.h"

namespace Signal {
namespace Processing {

class Invalidator
{
public:
    typedef boost::shared_ptr<Invalidator> Ptr;

    virtual ~Invalidator() {}

    virtual void deprecateCache(Step::Ptr at, Signal::Intervals what) const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_INVALIDATOR_H
