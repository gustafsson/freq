#include "targetmarker.h"
#include "dag.h"
#include "unused.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

TargetMarker::
        TargetMarker(TargetNeeds::ptr target_needs, Dag::ptr dag)
    :
      target_needs_(target_needs),
      dag_(dag)
{
    EXCEPTION_ASSERT(target_needs_);
    EXCEPTION_ASSERT(dag_);
}


std::set<Step::ptr> single_paths(GraphVertex v, const Graph& g) {
    std::set<Step::ptr> S;

    UNUSED(int od) = boost::out_degree(v, g);
    UNUSED(int id) = boost::in_degree(v, g);

    if (boost::out_degree(v, g) > 1)
        return S;

    S.insert (g[v]);

    BOOST_FOREACH(GraphEdge e, boost::in_edges(v, g)) {
        std::set<Step::ptr> s = single_paths(boost::source(e,g), g);
        S.insert (s.begin (), s.end ());
    }

    return S;
}


TargetMarker::
        ~TargetMarker()
{
    Step::ptr step = target_needs_.read ()->step().lock();
    if (!step)
        return;

    // Remove all steps than can only be reached from this target.
    auto dag = dag_.write ();
    GraphVertex start = dag->getVertex (step);
    if (!start)
        return;

    std::set<Step::ptr> steps_to_remove = single_paths(start, dag->g ());

    BOOST_FOREACH( Step::ptr s, steps_to_remove ) {
        dag->removeStep (s);
    }
}


shared_state<TargetNeeds> TargetMarker::
        target_needs() const
{
    return target_needs_;
}


Step::ptr::weak_ptr TargetMarker::
        step() const
{
    return target_needs_.read ()->step();
}

} // namespace Processing
} // namespace Signal

#include "bedroom.h"
#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

void TargetMarker::
        test()
{
    // It should mark the position of a target in the dag and remove it's vertices when the marker is deleted.
    {
        Step::ptr step1(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step2a(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step3a(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step2b(new Step(Signal::OperationDesc::ptr()));

        Bedroom::ptr bedroom(new Bedroom());
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        TargetNeeds::ptr target_needs(new TargetNeeds(step3a, notifier));
        Dag::ptr dagp(new Dag());

        TargetMarker::ptr tm( new TargetMarker(target_needs, dagp));

        {
            auto dag = dagp.write ();
            const Graph& g = dag->g ();
            dag->insertStep (step3a);
            dag->insertStep (step1, dag->getVertex (step3a));
            dag->insertStep (step2a, dag->getVertex (step3a));
            dag->appendStep (step2b, dag->getVertex (step1));
            EXCEPTION_ASSERT_EQUALS( g.num_edges (), 3u );
            EXCEPTION_ASSERT_EQUALS( g.num_vertices (), 4u );
        }

        tm.reset ();

        {
            auto dag = dagp.write ();
            const Graph& g = dag->g ();
            EXCEPTION_ASSERT_EQUALS( g.num_edges (), 1u );
            EXCEPTION_ASSERT_EQUALS( g.num_vertices (), 2u );

            EXCEPTION_ASSERT_EQUALS( boost::out_degree(dag->getVertex (step1), g), 1u );
            EXCEPTION_ASSERT_EQUALS( boost::in_degree(dag->getVertex (step1), g), 0u );
            EXCEPTION_ASSERT_EQUALS( boost::out_degree(dag->getVertex (step2b), g), 0u );
            EXCEPTION_ASSERT_EQUALS( boost::in_degree(dag->getVertex (step2b), g), 1u );
        }
    }
}


} // namespace Processing
} // namespace Signal
