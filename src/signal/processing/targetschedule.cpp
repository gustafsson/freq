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
        TargetSchedule(Dag::ptr g, IScheduleAlgorithm::ptr algorithm, Targets::ptr targets)
    :
      targets(targets),
      g(g),
      algorithm(algorithm)
{
    BOOST_ASSERT(g);
    BOOST_ASSERT(algorithm);
}


Task::ptr TargetSchedule::
        getTask(Signal::ComputingEngine::ptr engine) const
{
    // Lock this from writing during getTask
    // Lock the graph from writing during getTask
    auto dag = g.read();

    TargetNeeds::ptr::read_ptr target = prioritizedTarget();
    if (!target) {
        DEBUGINFO TaskInfo("No target needs anything right now");
        return Task::ptr();
    }

    Step::ptr step                              = target->step().lock ();
    Signal::Intervals needed                    = target->not_started();
    Signal::IntervalType work_center            = target->work_center();
    Signal::IntervalType preferred_update_size  = target->preferred_update_size();

    target.unlock (); // release lock on TargetNeeds

    DEBUGINFO TaskTimer tt(boost::format("getTask(%s,%g)") % needed % work_center);

    EXCEPTION_ASSERT(step);
    GraphVertex vertex = dag->getVertex(step);
    EXCEPTION_ASSERT(vertex);

    Task::ptr task = algorithm.read ()->getTask(
            dag->g(),
            vertex,
            needed,
            work_center,
            preferred_update_size,
            Workers::ptr(),
            engine);

    DEBUGINFO if (task)
        TaskInfo(boost::format("task->expected_output() = %s") % task.read ()->expected_output());

    return task;
}


TargetNeeds::ptr::read_ptr TargetSchedule::
        prioritizedTarget() const
{
    TargetNeeds::ptr::read_ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(const TargetNeeds::ptr& t, targets.read ()->getTargets())
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
    virtual Task::ptr getTask(
            const Graph&,
            GraphVertex,
            Signal::Intervals needed,
            Signal::IntervalType,
            Signal::IntervalType,
            Workers::ptr,
            Signal::ComputingEngine::ptr) const
    {
        return Task::ptr(new Task(Step::ptr(new Step(Signal::OperationDesc::ptr())).write(),
                                  Step::ptr(),
                                  std::vector<Step::const_ptr>(),
                                  Signal::Operation::ptr(),
                                  needed.spannedInterval (),
                                  Signal::Interval()));
    }
};


void TargetSchedule::
        test()
{
    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Dag::ptr dag(new Dag);
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        dag.write ()->appendStep(step);
        IScheduleAlgorithm::ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        Targets::ptr targets(new Targets(notifier));
        Signal::ComputingEngine::ptr engine;

        TargetSchedule targetschedule(dag, algorithm, targets);

        // It should not return a task without a target
        EXCEPTION_ASSERT(!targetschedule.getTask (engine));

        // It should not return a task for a target without needed_samples
        TargetNeeds::ptr targetneeds ( targets.write ()->addTarget(step) );
        EXCEPTION_ASSERT(!targetschedule.getTask (engine));

        // The scheduler should be used to find a task when the target has
        // a non-empty not_started();
        targetneeds.write ()->updateNeeds(Signal::Interval(3,4));
        EXCEPTION_ASSERT(targetneeds.read ()->not_started());
        Task::ptr task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.read ()->expected_output(), Signal::Interval(3,4));
    }

    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        Targets::ptr targets(new Targets(notifier));
    }

    // It should work on less prioritized tasks if the high prio tasks are done
    {
        Dag::ptr dag(new Dag);
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step2(new Step(Signal::OperationDesc::ptr()));
        dag.write ()->appendStep(step);
        dag.write ()->appendStep(step2); // same dag object, but not connected
        IScheduleAlgorithm::ptr algorithm(new GetDagTaskAlgorithmMockup);
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        Targets::ptr targets(new Targets(notifier));
        Signal::ComputingEngine::ptr engine;

        TargetNeeds::ptr targetneeds ( targets.write ()->addTarget(step) );
        TargetNeeds::ptr targetneeds2 ( targets.write ()->addTarget(step2) );
        targetneeds.write ()->updateNeeds(Signal::Interval(),0,10,1);
        targetneeds2.write ()->updateNeeds(Signal::Interval(5,6),0,10,0);

        TargetSchedule targetschedule(dag, algorithm, targets);
        Task::ptr task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.read ()->expected_output(), Signal::Interval(5,6));
    }
}


} // namespace Processing
} // namespace Signal
