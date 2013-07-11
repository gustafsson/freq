// Include Boost.Foreach before any Qt includes to prevent conflicts with Qt foreach
#include <boost/foreach.hpp>

#include "schedulegettask.h"
#include "schedulealgorithm.h"

using namespace boost::posix_time;

namespace Signal {
namespace Processing {


ScheduleGetTask::
        ScheduleGetTask(Dag::Ptr g)
    :
      g(g)
{
}


Task::Ptr ScheduleGetTask::
        getTask() volatile
{
    WritePtr selftask(this);
    return dynamic_cast<ScheduleGetTask*>((GetTask*)selftask)->getTask ();
}


Task::Ptr ScheduleGetTask::
        getTask()
{
    ScheduleAlgorithm sa;

    Target::Ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(Target::Ptr t, write1(g)->target)
    {
        ptime last_request = read1(t)->last_request();

        if (latest < last_request) {
            latest = last_request;
            target = t;
        }
    }

    if (!target)
        return Task::Ptr();

    Step::Ptr step = read1(target)->step();
    GraphVertex vertex = write1(g)->map[step];

    Signal::Intervals missing_in_target = write1(target)->out_of_date();
    Signal::IntervalType work_center = read1(target)->work_center();

    Task::Ptr task = sa.getTask(
            write1(g)->g,
            vertex,
            missing_in_target,
            work_center);

    return task;
}


} // namespace Processing
} // namespace Signal
