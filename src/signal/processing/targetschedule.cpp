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


Task TargetSchedule::
        getTask(Signal::ComputingEngine::ptr engine) const
{
    // Lock this from writing during getTask
    // Lock the graph from writing during getTask
    auto dag = g.read();

    TargetState targetstate = prioritizedTarget();
    TargetNeeds::State& state = targetstate.second;
    Step::ptr& step = targetstate.first;
    if (!step) {
        DEBUGINFO TaskInfo("No target needs anything right now");
        return Task();
    }

    DEBUGINFO TaskTimer tt(boost::format("getTask(%s,%g)") % state.needed_samples % state.work_center);

    GraphVertex vertex = dag->getVertex(step);
    EXCEPTION_ASSERT(vertex);

    Task task = algorithm.read ()->getTask(
            dag->g(),
            vertex,
            state.needed_samples,
            state.work_center,
            state.preferred_update_size,
            Workers::ptr(),
            engine);

    DEBUGINFO if (task)
        TaskInfo(boost::format("task->expected_output() = %s") % task.expected_output());

    return task;
}


TargetSchedule::TargetState TargetSchedule::
        prioritizedTarget() const
{
    TargetState r;

    r.second.last_request = neg_infin;

    for (const TargetNeeds::ptr& t: targets->getTargets())
    {
        auto step = t->step ().lock ();
        if (!step)
            continue;

        Signal::Intervals step_needed = step.read()->not_started();
        if (!step_needed)
            continue;

        TargetNeeds::State state = t->state ();
        state.needed_samples &= step_needed;

        if (r.second.last_request < state.last_request && state.needed_samples)
        {
            r.first = step;
            r.second = state;
        }
    }

    return r;
}

} // namespace Processing
} // namespace Signal

#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

class GetDagTaskAlgorithmMockup: public IScheduleAlgorithm
{
public:
    virtual Task getTask(
            const Graph&,
            GraphVertex,
            Signal::Intervals needed,
            Signal::IntervalType,
            Signal::IntervalType,
            Workers::ptr,
            Signal::ComputingEngine::ptr) const
    {
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        return Task(step.write(), step,
                                  std::vector<Step::const_ptr>(),
                                  Signal::Operation::ptr(),
                                  needed.spannedInterval (),
                                  Signal::Interval());
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
        TargetNeeds::ptr targetneeds ( targets->addTarget(step) );
        EXCEPTION_ASSERT(!targetschedule.getTask (engine));

        // The scheduler should be used to find a task when the target has
        // a non-empty not_started();
        targetneeds->updateNeeds(Signal::Interval(3,4));
        EXCEPTION_ASSERT(targetneeds->not_started());
        Task task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.expected_output(), Signal::Interval(3,4));
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

        TargetNeeds::ptr targetneeds ( targets->addTarget(step) );
        TargetNeeds::ptr targetneeds2 ( targets->addTarget(step2) );
        targetneeds->updateNeeds(Signal::Interval(),0,10,1);
        targetneeds2->updateNeeds(Signal::Interval(5,6),0,10,0);

        TargetSchedule targetschedule(dag, algorithm, targets);
        Task task = targetschedule.getTask (engine);
        EXCEPTION_ASSERT(task);
        EXCEPTION_ASSERT_EQUALS(task.expected_output(), Signal::Interval(5,6));
    }
}


} // namespace Processing
} // namespace Signal
