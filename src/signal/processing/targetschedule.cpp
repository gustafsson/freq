// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "targetschedule.h"

//#define DEBUGINFO
#define DEBUGINFO if(0)

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
    const TargetSchedule* self = (const TargetSchedule*)&*gettask;

    // Lock the graph from writing during getTask
    Dag::ReadPtr dag(self->g);

    TargetNeeds::Ptr priotarget = self->prioritizedTarget();
    if (!priotarget)
        return Task::Ptr();

    Step::Ptr step;
    Signal::Intervals needed;
    Signal::IntervalType work_center;
    Signal::IntervalType preferred_update_size;

    // Read info from target
    {
        TargetNeeds::ReadPtr target(priotarget);
        step = target->step().lock ();
        needed = target->not_started();
        work_center = target->work_center();
        preferred_update_size = target->preferred_update_size();
    }

    if (!needed || !step)
        return Task::Ptr();

    GraphVertex vertex = dag->getVertex(step);
    if (!vertex)
        return Task::Ptr();

    DEBUGINFO TaskTimer tt(boost::format("getTask(%s,%g)") % needed % work_center);

    Task::Ptr task = read1(self->algorithm)->getTask(
            dag->g(),
            vertex,
            needed,
            work_center,
            preferred_update_size);

    DEBUGINFO if (task)
        TaskInfo(boost::format("task->expected_output() = %s") % read1(task)->expected_output());

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
        Dag::Ptr dag(new Dag);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        write1(dag)->appendStep(step);
        IScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::Ptr bedroom(new Bedroom);
        Targets::Ptr targets(new Targets(bedroom));

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
