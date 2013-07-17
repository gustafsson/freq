// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "targetschedule.h"

using namespace boost::posix_time;

namespace Signal {
namespace Processing {


TargetSchedule::
        TargetSchedule(Dag::Ptr g, IScheduleAlgorithm::Ptr algorithm, Targets::Ptr targets)
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
    const TargetSchedule* self = dynamic_cast<const TargetSchedule*>((const ISchedule*)gettask);

    // Lock the graph from writing during getTask
    Dag::ReadPtr dag(self->g);

    TargetNeeds::Ptr priotarget = self->prioritizedTarget();
    if (!priotarget)
        return Task::Ptr();

    Step::Ptr step;
    Signal::Intervals missing_in_target;
    Signal::IntervalType work_center;

    // Read info from target
    {
        TargetNeeds::ReadPtr target(priotarget);
        step = target->step();
        missing_in_target = target->not_started();
        work_center = target->work_center();
    }

    if (!missing_in_target)
        return Task::Ptr();

    GraphVertex vertex = dag->getVertex(step);

    Task::Ptr task = read1(self->algorithm)->getTask(
            dag->g(),
            vertex,
            missing_in_target,
            work_center);

    return task;
}


TargetNeeds::Ptr TargetSchedule::
        prioritizedTarget() const
{
    TargetNeeds::Ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(const TargetNeeds::Ptr& t, read1(targets)->getTargets())
    {
        ptime last_request = read1(t)->last_request();

        if (latest < last_request) {
            latest = last_request;
            target = t;
        }
    }

    return target;
}


class GetDagTaskAlgorithmMockup: public IScheduleAlgorithm
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
        return Task::Ptr(new Task(0, Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(5,6)));
    }
};


void TargetSchedule::
        test()
{
    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        Dag::Ptr dag(new Dag);
        write1(dag)->appendStep(step);
        IScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Targets::Ptr targets(new Targets(Bedroom::Ptr(new Bedroom)));

        TargetSchedule targetschedule(dag, algorithm, targets);

        // It should not return a task without a target
        EXCEPTION_ASSERT(!targetschedule.getTask ());

        // It should not return a task for a target without needed_samples
        TargetNeeds::Ptr targetneeds ( write1(targets)->addTarget(step) );
        EXCEPTION_ASSERT(!targetschedule.getTask ());

        // The scheduler should be used to find a task when the target has
        // a non-empty not_started();
        write1(targetneeds)->updateNeeds(Signal::Interval(3,4));
        EXCEPTION_ASSERT(read1(targetneeds)->not_started());
        Task::Ptr task = targetschedule.getTask ();
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(read1(task)->expected_output(), Signal::Interval(5,6));
    }

    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Targets::Ptr targets(new Targets(Bedroom::Ptr(new Bedroom)));
    }
}


} // namespace Processing
} // namespace Signal
