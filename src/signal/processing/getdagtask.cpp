// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "getdagtask.h"

using namespace boost::posix_time;

namespace Signal {
namespace Processing {


GetDagTask::
        GetDagTask(Dag::Ptr g, GetDagTaskAlgorithm::Ptr algorithm)
    :
      g(g),
      algorithm(algorithm)
{
    BOOST_ASSERT(g);
    BOOST_ASSERT(algorithm);
}


Task::Ptr GetDagTask::
        getTask() volatile
{
    // Lock this from writing during getTask
    ReadPtr gettask(this);
    const GetDagTask* self = dynamic_cast<const GetDagTask*>((const GetTask*)gettask);

    // Lock the graph from writing during getTask
    Dag::ReadPtr dag(self->g);

    Target::Ptr priotarget = self->prioritizedTarget();
    if (!priotarget)
        return Task::Ptr();

    Step::Ptr step;
    Signal::Intervals missing_in_target;
    Signal::IntervalType work_center;

    // Read info from target
    {
        Target::WritePtr target(priotarget);
        step = target->step();
        missing_in_target = target->out_of_date();
        work_center = target->work_center();
    }

    GraphVertex vertex = dag->getVertex(step);

    Task::Ptr task = read1(self->algorithm)->getTask(
            dag->g(),
            vertex,
            missing_in_target,
            work_center);

    return task;
}


Target::Ptr GetDagTask::
        prioritizedTarget() const
{
    Target::Ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(Target::Ptr t, this->targets)
    {
        ptime last_request = read1(t)->last_request();

        if (latest < last_request) {
            latest = last_request;
            target = t;
        }
    }

    return target;
}


} // namespace Processing
} // namespace Signal
