#include <QObject>
#include "graphupdater.h"
#include "schedulegettask.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphUpdater::
        GraphUpdater(Dag::Ptr dag, GetTask::Ptr scheduleGetTask)
    :
      dag_(dag),
      schedule_get_task_(scheduleGetTask)
{
    EXCEPTION_ASSERT(schedule_get_task_);
    EXCEPTION_ASSERT(dynamic_cast<const ScheduleGetTask*>(&*read1(schedule_get_task_)));
}


void GraphUpdater::
        deprecateCache(Step::Ptr s, Signal::Intervals /*what*/) const
{
    deprecateCache(Dag::ReadPtr(dag_), s);

    ScheduleGetTask* t = dynamic_cast<ScheduleGetTask*>(&*write1(schedule_get_task_));
    t->wakeup ();
}


void GraphUpdater::
        deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const
{
    write1(s)->deprecateCache(Signal::Intervals::Intervals_ALL);

    BOOST_FOREACH(Step::Ptr ts, dag->targetSteps(s)) {
        deprecateCache(dag, ts);
    }
}

} // namespace Processing
} // namespace Signal
