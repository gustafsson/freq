#include <QObject>

#include "dag.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/foreach.hpp>

namespace og = boost::graph;

namespace Signal {
namespace Processing {

Dag::
        Dag()
{
}


GraphVertex Dag::
        getVertex(Step::Ptr s) const
{
    StepVertexMap::const_iterator i = map.find (s);
    EXCEPTION_ASSERTX (i != map.end (), "step was not found");

    return i->second;
}


GraphVertex Dag::
        appendStep(Step::Ptr step, GraphVertex v)
{
    EXCEPTION_ASSERT (step);
    StepVertexMap::const_iterator i = map.find (step);
    EXCEPTION_ASSERTX (i == map.end (), "step was already added");

    GraphVertex new_vertex = g_.add_vertex (step);
    map[step] = new_vertex;

    if (v != boost::graph_traits<Graph>::null_vertex ()) {
        g_.add_edge (v, new_vertex);
    }

    return new_vertex;
}


void Dag::
        removeStep(Step::Ptr step)
{
    EXCEPTION_ASSERT (step);

    GraphVertex v = getVertex(step);
    int id = in_degree(v, g_);
    int od = out_degree(v, g_);

    if (id <= 1) {
        if (id == 1) {
            GraphEdge e = *in_edges(v, g_).first;
            GraphVertex singlesource = source(e, g_);

            std::vector<GraphVertex> targets;
            BOOST_FOREACH(GraphEdge e, out_edges(v, g_)) {
                GraphVertex u = boost::target(e,g_);
                targets.push_back (u);
            }

            BOOST_FOREACH(GraphVertex u, targets) {
                g_.add_edge (singlesource, u);
            }
        }
    } else if (od <= 1) {
        if (od == 1) {
            GraphEdge e = *out_edges(v, g_).first;
            GraphVertex singletarget = boost::target(e, g_);

            std::vector<GraphVertex> sources;
            BOOST_FOREACH(GraphEdge e, in_edges(v, g_)) {
                sources.push_back (source(e,g_));
            }

            BOOST_FOREACH(GraphVertex u, sources) {
                g_.add_edge (u, singletarget);
            }
        }
    } else {
        EXCEPTION_ASSERTX (id <= 1 || od <= 1, "Can't remove a step with both multiple sources and multiple targets");
    }

    g_.clear_vertex (v);
    g_.remove_vertex (v);
    map.erase (step);
}


std::vector<Step::Ptr> Dag::
        sourceSteps(Step::Ptr step) const
{
    std::vector<Step::Ptr> steps;

    BOOST_FOREACH(GraphEdge e, in_edges(getVertex(step), g_)) {
        GraphVertex u = source(e,g_);
        steps.push_back (g_[u]);
    }

    return steps;
}


std::vector<Step::Ptr> Dag::
        targetSteps(Step::Ptr step) const
{
    std::vector<Step::Ptr> steps;

    BOOST_FOREACH(GraphEdge e, out_edges(getVertex(step), g_)) {
        GraphVertex u = boost::target(e,g_);
        Step::Ptr s = g_[u];
        //int nc = write1(s)->num_channels ();
        steps.push_back (s);
    }

    return steps;
}


void Dag::
        test()
{
    // It should manage the connections between the steps in the signal processing chain.
    {
        Dag dag;

        Step::Ptr step1(new Step(Signal::OperationDesc::Ptr(), 1, 1));
        Step::Ptr step2(new Step(Signal::OperationDesc::Ptr(), 1, 2));
        Step::Ptr step3(new Step(Signal::OperationDesc::Ptr(), 1, 3));

        dag.appendStep (step1);
        dag.appendStep (step2, dag.getVertex (step1));
        dag.appendStep (step3, dag.getVertex (step2));

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::Ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::Ptr>(1, step2));
        EXCEPTION_ASSERT(dag.sourceSteps (step2) == std::vector<Step::Ptr>(1, step1));
        EXCEPTION_ASSERT(dag.targetSteps (step2) == std::vector<Step::Ptr>(1, step3));
        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::Ptr>(1, step2));
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::Ptr>());

        dag.removeStep (step2);

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::Ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::Ptr>(1, step3));
        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::Ptr>(1, step1));
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::Ptr>());

        dag.removeStep (step3);

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::Ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::Ptr>());

        dag.appendStep (step2, dag.getVertex (step1));

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::Ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::Ptr>(1, step2));
        EXCEPTION_ASSERT(dag.sourceSteps (step2) == std::vector<Step::Ptr>(1, step1));
        EXCEPTION_ASSERT(dag.targetSteps (step2) == std::vector<Step::Ptr>());
    }
}


} // namespace Processing
} // namespace Signal
