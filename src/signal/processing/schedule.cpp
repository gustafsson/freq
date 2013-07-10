#include "schedule.h"

#include "schedulealgorithm.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {


Schedule::
        Schedule(Dag::Ptr g)
    :
      g(g)
{
}


Task::Ptr Schedule::
        getTask()
{
    ScheduleAlgorithm sa;

    Target::Ptr target;

    BOOST_FOREACH(Target::Ptr t, write1(g)->target) {
        target = t;
    }

    if (!target)
        return Task::Ptr();

    Step::Ptr step = write1(target)->step();
    GraphVertex vertex = write1(g)->map[step];

    Signal::Intervals missing_in_target = write1(target)->out_of_date();
    int preferred_size = 1 + missing_in_target.count () / workers.size ();
    int center = target->center;
    center = Interval::IntervalType_MIN;

    Task::Ptr task = sa.getTask(
            write1(g)->g,
            vertex,
            missing_in_target,
            preferred_size,
            center);

    return task;
}


void Schedule::
        test()
{

}


} // namespace Processing
} // namespace Signal
