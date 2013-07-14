#ifndef SIGNAL_PROCESSING_UPDATER_H
#define SIGNAL_PROCESSING_UPDATER_H

#include "signal/intervals.h"
#include <boost/shared_ptr.hpp>

namespace Signal {
namespace Processing {

class Updater
{
public:
    typedef boost::shared_ptr<Updater> Ptr;

    virtual ~Updater() {}

    virtual void update(int prio, Signal::IntervalType center, Signal::Intervals intervals) = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_UPDATER_H
