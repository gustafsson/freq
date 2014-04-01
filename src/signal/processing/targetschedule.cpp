// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "targetschedule.h"
#include "tasktimer.h"

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
        getTask(Signal::ComputingEngine::Ptr engine) const
{
    // Lock this from writing during getTask
    // Lock the graph from writing during getTask
    auto dag = g.read();

    TargetNeeds::Ptr::read_ptr target = prioritizedTarget();
    if (!target) {
        DEBUGINFO TaskInfo("No target needs anything right now");
        return Task::Ptr();
    }

    Step::Ptr step                              = target->step().lock ();
    Signal::Intervals needed                    = target->not_started();
    Signal::IntervalType work_center            = target->work_center();
    Signal::IntervalType preferred_update_size  = target->preferred_update_size();

    target.unlock (); // release lock on TargetNeeds

    DEBUGINFO TaskTimer tt(boost::format("getTask(%s,%g)") % needed % work_center);

    EXCEPTION_ASSERT(step);
    GraphVertex vertex = dag->getVertex(step);
    EXCEPTION_ASSERT(vertex);

    Task::Ptr task = algorithm.read ()->getTask(
            dag->g(),
            vertex,
            needed,
            work_center,
            preferred_update_size,
            Workers::Ptr(),
            engine);

    DEBUGINFO if (task)
        TaskInfo(boost::format("task->expected_output() = %s") % task.read ()->expected_output());

    return task;
}


TargetNeeds::Ptr::read_ptr TargetSchedule::
        prioritizedTarget() const
{
    TargetNeeds::Ptr::read_ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(const TargetNeeds::Ptr& t, targets.read ()->getTargets())
    {
        auto rt = t.read ();

        Signal::Intervals needed = rt->not_started();
        ptime last_request       = rt->last_request();

        if (latest < last_request && needed) {
            latest = last_request;
            target.swap (rt);
        }
    }

    return target;
}

} // namespace Processing
} // namespace Signal

#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

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
        return Task::Ptr(new Task(Step::Ptr(new Step(Signal::OperationDesc::Ptr())).write(),
                                  Step::Ptr(),
                                  std::vector<Step::Ptr>(),
                                  Signal::Operation::Ptr(),
                                  needed.spannedInterval (),
                                  Signal::Interval()));
    }
};


void TargetSchedule::
        test()
{
    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Dag::Ptr dag(new Dag);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        dag.write ()->appendStep(step);
        IScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::Ptr bedroom(new Bedroom);
        BedroomNotifier::Ptr notifier(new BedroomNotifier(bedroom));
        Targets::Ptr targets(new Targets(notifier));
        Signal::ComputingEngine::Ptr engine;

        TargetSchedule targetschedule(dag, algorithm, targets);

        // It should not return a task without a target
        EXCEPTION_ASSERT(!targetschedule.getTask (engine));

        // It should not return a task for a target without needed_samples
        TargetNeeds::Ptr targetneeds ( targets.write ()->addTarget(step) );
        EXCEPTION_ASSERT(!targetschedule.getTask (engine));

        // The scheduler should be used to find a task when the target has
        // a non-empty not_started();
        targetneeds.write ()->updateNeeds(Signal::Interval(3,4));
        EXCEPTION_ASSERT(targetneeds.read ()->not_started());
        Task::Ptr task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.read ()->expected_output(), Signal::Interval(3,4));
    }

    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Bedroom::Ptr bedroom(new Bedroom);
        BedroomNotifier::Ptr notifier(new BedroomNotifier(bedroom));
        Targets::Ptr targets(new Targets(notifier));
    }

    // It should work on less prioritized tasks if the high prio tasks are done
    {
        Dag::Ptr dag(new Dag);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        Step::Ptr step2(new Step(Signal::OperationDesc::Ptr()));
        dag.write ()->appendStep(step);
        dag.write ()->appendStep(step2); // same dag object, but not connected
        IScheduleAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::Ptr bedroom(new Bedroom);
        BedroomNotifier::Ptr notifier(new BedroomNotifier(bedroom));
        Targets::Ptr targets(new Targets(notifier));
        Signal::ComputingEngine::Ptr engine;

        TargetNeeds::Ptr targetneeds ( targets.write ()->addTarget(step) );
        TargetNeeds::Ptr targetneeds2 ( targets.write ()->addTarget(step2) );
        targetneeds.write ()->updateNeeds(Signal::Interval(),0,10,1);
        targetneeds2.write ()->updateNeeds(Signal::Interval(5,6),0,10,0);

        TargetSchedule targetschedule(dag, algorithm, targets);
        Task::Ptr task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.read ()->expected_output(), Signal::Interval(5,6));
    }
}


} // namespace Processing
} // namespace Signal
