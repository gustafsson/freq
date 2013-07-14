#ifndef SIGNAL_PROCESSING_TARGETUPDATER_H
#define SIGNAL_PROCESSING_TARGETUPDATER_H

#include "signal/intervals.h"
#include <boost/shared_ptr.hpp>

namespace Signal {
namespace Processing {

class TargetUpdater
{
public:
    typedef boost::shared_ptr<TargetUpdater> Ptr;

    virtual ~TargetUpdater() {}

    virtual void update(int prio, Signal::IntervalType center, Signal::Intervals intervals) = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETUPDATER_H
