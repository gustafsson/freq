#ifndef SIGNAL_PROCESSING_GRAPHUPDATER_H
#define SIGNAL_PROCESSING_GRAPHUPDATER_H

#include "dag.h"
#include "gettask.h"
#include "invalidator.h"

namespace Signal {
namespace Processing {

class GraphUpdater: public Invalidator
{
public:
    GraphUpdater(Dag::Ptr dag, GetTask::Ptr scheduleGetTask);

    void deprecateCache(Step::Ptr s, Signal::Intervals what) const;

private:
    // invalidate steps (only deprecateCache(Interval::Interval_ALL) until OperationDesc supports 'affectedInterval' (inverse of requiredInterval))
    void deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const;

    Dag::Ptr dag_;
    GetTask::Ptr schedule_get_task_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GRAPHUPDATER_H
