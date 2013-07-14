// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "getdagtask.h"

using namespace boost::posix_time;

namespace Signal {
namespace Processing {


GetDagTask::
        GetDagTask(Dag::Ptr g, GetDagTaskAlgorithm::Ptr algorithm, std::vector<Target::Ptr> targets)
    :
      targets(targets),
      g(g),
      algorithm(algorithm)
{
    BOOST_ASSERT(g);
    BOOST_ASSERT(algorithm);
}


TargetUpdater::Ptr GetDagTask::
        addTarget(Step::Ptr step)
{
    Invalidator::Ptr invalidator();
    Target::Ptr target(new Target(step));


    //TargetUpdater::Ptr targetUpdater(new TargetInvalidator(invalidator, target);
    return TargetUpdater::Ptr();
}


void GetDagTask::
        removeTarget(Step::Ptr step)
{
    for (TargetInfos::iterator i=targets.begin (); i!=targets.end (); i++) {
        Target::Ptr t = *i;
        if (read1(t)->step == step) {
            targets.erase (i);
            return;
        }
    }
}


std::vector<Step::Ptr> GetDagTask::
        getTargets() const
{
    std::vector<Step::Ptr> target_steps;

    BOOST_FOREACH( Target::Ptr t, targets ) {
        target_steps.push_back (read1(t)->step);
    }

    return target_steps;
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


Target::Ptr GetDagTask::
        prioritizedTarget() const
{
    Target::Ptr target;

    ptime latest(neg_infin);
    BOOST_FOREACH(Target::Ptr t, this->targets)
    {
        ptime last_request = read1(t)->last_request;

        if (latest < last_request) {
            latest = last_request;
            target = t;
        }
    }

    return target;
}


class GetDagTaskAlgorithmMockup: public GetDagTaskAlgorithm
{
public:
    virtual Task::Ptr getTask(
            const Graph&,
            GraphVertex,
            Signal::Intervals,
            Signal::IntervalType,
            Signal::ComputingEngine::Ptr) const
    {
        return Task::Ptr(new Task(Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(1,2)));
    }
};


void GetDagTask::
        test()
{
    // It should provide tasks to keep a Dag up-to-date with respect to all targets
    {
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));
        Dag::Ptr dag(new Dag);
        write1(dag)->appendStep(step);

        GetDagTaskAlgorithm::Ptr algorithm(new GetDagTaskAlgorithmMockup);
        std::vector<Target::Ptr> targets;
        //targets.push_back (Target::Ptr(new GetDagTask_TargetMockup(step)));
        GetDagTask getdagtask(dag, algorithm, targets);
        Task::Ptr task = getdagtask.getTask ();

        EXCEPTION_ASSERT(0 == task.get ()); // should not be null
        /*
        EXCEPTION_ASSERT_EQUALS(read1(task)->expected_output(), Signal::Interval(1,3));*/
    }
}


} // namespace Processing
} // namespace Signal
