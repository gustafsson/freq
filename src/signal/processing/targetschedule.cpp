// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "targetschedule.h"

using namespace boost::posix_time;

namespace Signal {
namespace Processing {


TargetSchedule::
        TargetSchedule(Dag::Ptr g, ScheduleAlgorithm::Ptr algorithm, Targets::Ptr targets)
    :
      targets(targets),
      g(g),
      algorithm(algorithm)
{
    BOOST_ASSERT(g);
    BOOST_ASSERT(algorithm);
}


Task::Ptr TargetSchedule::
        getTask() volatile
{
    // Lock this from writing during getTask
    ReadPtr gettask(this);
    const TargetSchedule* self = dynamic_cast<const TargetSchedule*>((const Schedule*)gettask);

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
        step = target->step;
        missing_in_target = read1(step)->not_started();
        work_center = target->work_center;
    }

    GraphVertex vertex = dag->getVertex(step);

    Task::Ptr task = read1(self->algorithm)->getTask(
            dag->g(),
            vertex,
            missing_in_target,
            work_center);

    return task;
}


Target::Ptr TargetSchedule::
        prioritizedTarget() const
{
    Target::Ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(Target::Ptr t, read1(targets)->getTargets())
    {
        ptime last_request = read1(t)->last_request;

        if (latest < last_request) {
            latest = last_request;
            target = t;
        }
    }

    return target;
}


class GetDagTaskAlgorithmMockup: public ScheduleAlgorithm
{
public:
    virtual Task::Ptr getTask(
            const Graph&,
            GraphVertex,
            Signal::Intervals,
            Signal::IntervalType,
            Workers::Ptr,
            Signal::ComputingEngine::Ptr) const
    {
        return Task::Ptr(new Task(Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(1,2)));
    }
};


void TargetSchedule::
        test()
{
    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));
        Dag::Ptr dag(new Dag);
        write1(dag)->appendStep(step);

        ScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Targets::Ptr targets(new Targets(dag, Bedroom::Ptr(new Bedroom)));
        //targets.push_back (Target::Ptr(new GetDagTask_TargetMockup(step)));
        TargetSchedule getdagtask(dag, algorithm, targets);
        Task::Ptr task = getdagtask.getTask ();

        EXCEPTION_ASSERT(0 == task.get ()); // should not be null
        /*
        EXCEPTION_ASSERT_EQUALS(read1(task)->expected_output(), Signal::Interval(1,3));*/
    }
}


} // namespace Processing
} // namespace Signal
