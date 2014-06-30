#include <QObject>

#include "dag.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/foreach.hpp>

namespace og = boost::graph;

namespace Signal {
namespace Processing {


GraphVertex
        NullVertex()
    {
    return boost::graph_traits<Graph>::null_vertex ();
    }


Dag::
        Dag()
{
}


GraphVertex Dag::
        getVertex(Step::ptr s) const
{
    StepVertexMap::const_iterator i = map.find (s);

    if (map.end () == i)
        return NullVertex ();

    return i->second;
}


GraphVertex Dag::
        appendStep(Step::ptr step, GraphVertex v)
{
    EXCEPTION_ASSERT (step);
    StepVertexMap::const_iterator i = map.find (step);
    EXCEPTION_ASSERTX (i == map.end (), "step was already added");

    GraphVertex new_vertex = g_.add_vertex (step);
    map[step] = new_vertex;

    if (v != NullVertex ()) {
        // v should have new_vertex as target
        g_.add_edge (v, new_vertex);
    }

    return new_vertex;
}


GraphVertex Dag::
        insertStep(Step::ptr step, GraphVertex v)
{
    EXCEPTION_ASSERT (step);
    StepVertexMap::const_iterator i = map.find (step);
    EXCEPTION_ASSERTX (i == map.end (), "step was already added");

    GraphVertex new_vertex = g_.add_vertex (step);
    map[step] = new_vertex;

    if (v != NullVertex ()) {
        // All sources of v should have new_vertex as target instead of v.
        // new_vertex should have v as target

        std::vector<GraphEdge> inedges;
        BOOST_FOREACH(GraphEdge e, in_edges(v, g_)) {
            inedges.push_back (e);
        }

        BOOST_FOREACH(GraphEdge e, inedges) {
            GraphVertex u = boost::source(e, g_);
            g_.remove_edge (e);
            g_.add_edge (u, new_vertex);
        }

        g_.add_edge (new_vertex, v);
    }

    return new_vertex;
}


void Dag::
        removeStep(Step::ptr step)
{
    EXCEPTION_ASSERT (step);

    GraphVertex v = getVertex(step);
    if (!v)
        return;

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
    g_.renumber_indices ();
    map.erase (step);
}


std::vector<Step::ptr> Dag::
        sourceSteps(Step::ptr step) const
{
    std::vector<Step::ptr> steps;

    GraphVertex v = getVertex(step);
    if (!v)
        return steps;

    BOOST_FOREACH(GraphEdge e, in_edges(v, g_)) {
        GraphVertex u = source(e,g_);
        steps.push_back (g_[u]);
    }

    return steps;
}


std::vector<Step::ptr> Dag::
        targetSteps(Step::ptr step) const
{

    GraphVertex v = getVertex(step);
    if (!v)
        return std::vector<Step::ptr>();

    std::vector<Step::ptr> steps;
    steps.reserve (out_degree(v, g_));

    BOOST_FOREACH(GraphEdge e, out_edges(v, g_)) {
        GraphVertex u = boost::target(e,g_);
        Step::ptr s = g_[u];
        //int nc = s.write ()->num_channels ();
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

        Step::ptr step1(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step2(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step3(new Step(Signal::OperationDesc::ptr()));

        dag.insertStep (step2);
        dag.insertStep (step1, dag.getVertex (step2));
        dag.appendStep (step3, dag.getVertex (step2));

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::ptr>(1, step2));
        EXCEPTION_ASSERT(dag.sourceSteps (step2) == std::vector<Step::ptr>(1, step1));
        EXCEPTION_ASSERT(dag.targetSteps (step2) == std::vector<Step::ptr>(1, step3));
        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::ptr>(1, step2));
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::ptr>());

        dag.removeStep (step2);

        EXCEPTION_ASSERT(dag.sourceSteps (step1) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step1) == std::vector<Step::ptr>(1, step3));
        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::ptr>(1, step1));
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::ptr>());

        dag.removeStep (step1);

        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::ptr>());

        dag.insertStep (step2, dag.getVertex (step3));

        EXCEPTION_ASSERT(dag.sourceSteps (step2) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT(dag.targetSteps (step2) == std::vector<Step::ptr>(1, step3));
        EXCEPTION_ASSERT(dag.sourceSteps (step3) == std::vector<Step::ptr>(1, step2));
        EXCEPTION_ASSERT(dag.targetSteps (step3) == std::vector<Step::ptr>());
    }

    // It should treat Step's that aren't a part of the Dag as lonely islands.
    {
        Dag dag;

        Step::ptr step1(new Step(Signal::OperationDesc::ptr()));
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));

        dag.appendStep (step1);

        dag.removeStep (step);
        EXCEPTION_ASSERT( dag.sourceSteps (step) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT( dag.targetSteps (step) == std::vector<Step::ptr>());
        EXCEPTION_ASSERT( dag.getVertex (step) == NullVertex() );
        EXCEPTION_ASSERT_EQUALS (dag.g ().num_vertices (), 1u );
    }
}


} // namespace Processing
} // namespace Signal
