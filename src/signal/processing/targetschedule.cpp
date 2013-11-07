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

    boost::shared_ptr<TargetNeeds::ReadPtr> target = self->prioritizedTarget();
    if (!target) {
        DEBUGINFO TaskInfo("!target");
        return Task::Ptr();
    }

    Step::Ptr step                              = target->get ()->step().lock ();
    Signal::Intervals needed                    = target->get ()->not_started();
    Signal::IntervalType work_center            = target->get ()->work_center();
    Signal::IntervalType preferred_update_size  = target->get ()->preferred_update_size();

    target.reset (); // release lock on TargetNeeds

    DEBUGINFO TaskTimer tt(boost::format("getTask(%s,%g)") % needed % work_center);

    EXCEPTION_ASSERT(step);
    GraphVertex vertex = dag->getVertex(step);
    EXCEPTION_ASSERT(vertex);

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


boost::shared_ptr<TargetNeeds::ReadPtr> TargetSchedule::
        prioritizedTarget() const
{
    boost::shared_ptr<TargetNeeds::ReadPtr> target;

    ptime latest(neg_infin);
    BOOST_FOREACH(const TargetNeeds::Ptr& t, read1(targets)->getTargets())
    {
        boost::shared_ptr<TargetNeeds::ReadPtr> rt(new TargetNeeds::ReadPtr(t));

        Signal::Intervals needed = rt->get ()->not_started();
        ptime last_request       = rt->get ()->last_request();

        if (latest < last_request && needed) {
            latest = last_request;
            target = rt;
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
            Signal::Intervals needed,
            Signal::IntervalType,
            Signal::IntervalType,
            Workers::Ptr,
            Signal::ComputingEngine::Ptr) const
    {
        return Task::Ptr(new Task(0, Step::Ptr(), std::vector<Step::Ptr>(), needed.spannedInterval ()));
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
        EXCEPTION_ASSERT_EQUALS(read1(task)->expected_output(), Signal::Interval(3,4));
    }

    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Targets::Ptr targets(new Targets(Bedroom::Ptr(new Bedroom)));
    }

    // It should work on less prioritized tasks if the high prio tasks are done
    {
        Dag::Ptr dag(new Dag);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        Step::Ptr step2(new Step(Signal::OperationDesc::Ptr()));
        write1(dag)->appendStep(step);
        write1(dag)->appendStep(step2); // same dag object, but not connected
        IScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::Ptr bedroom(new Bedroom);
        Targets::Ptr targets(new Targets(bedroom));

        TargetNeeds::Ptr targetneeds ( write1(targets)->addTarget(step) );
        TargetNeeds::Ptr targetneeds2 ( write1(targets)->addTarget(step2) );
        write1(targetneeds)->updateNeeds(Signal::Interval(),0,10,Signal::Interval(),1);
        write1(targetneeds2)->updateNeeds(Signal::Interval(5,6),0,10,Signal::Interval(),0);

        TargetSchedule targetschedule(dag, algorithm, targets);
        Task::Ptr task = targetschedule.getTask ();
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(read1(task)->expected_output(), Signal::Interval(5,6));
    }
}


} // namespace Processing
} // namespace Signal
