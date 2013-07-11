// Include Boost.Foreach before any Qt includes to prevent conflicts with Qt foreach
#include <boost/foreach.hpp>

#include "schedulegettask.h"
#include "schedulealgorithm.h"
#include <QThread>

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
    Task::Ptr task;

    while (true) {
        {
            // Lock this while reading a task
            ReadPtr that(this);
            const ScheduleGetTask* self = dynamic_cast<const ScheduleGetTask*>((const GetTask*)that);
            task = self->getTask ();
        }

        if (task)
            return task;

        // QWaitCondition/QMutex are thread-safe so we can discard the volatile qualifier
        const_cast<QWaitCondition*>(&work_condition)->wait (
                    const_cast<QMutex*> (&work_condition_mutex));
    }
}


Task::Ptr ScheduleGetTask::
        getTask() const
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
    GraphVertex vertex = write1(g)->getVertex(step);

    Signal::Intervals missing_in_target = write1(target)->out_of_date();
    Signal::IntervalType work_center = read1(target)->work_center();

    Task::Ptr task = sa.getTask(
            read1(g)->g(), // this locks the graph for writing during getTask
            vertex,
            missing_in_target,
            work_center);

    return task;
}


void ScheduleGetTask::
        test()
{
    // It should provide new tasks for workers who lack information about what they should do
    {

    }

    // It should halt works while waiting for an available task
    {

    }
}


} // namespace Processing
} // namespace Signal
