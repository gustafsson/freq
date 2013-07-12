#ifndef SIGNAL_PROCESSING_DAG_H
#define SIGNAL_PROCESSING_DAG_H

#include <boost/graph/directed_graph.hpp>

#include "target.h"
#include "step.h"
#include "volatileptr.h"

namespace Signal {
namespace Processing {

typedef boost::directed_graph<Signal::Processing::Step::Ptr, boost::vecS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor GraphVertex;
typedef boost::graph_traits<Graph>::edge_descriptor GraphEdge;


/**
 * @brief The Dag class should manage the connections between the steps in the signal processing chain.
 *
 * example:
 *
 *  Graph g;
 *  Signal::Processing::Step::Ptr source, target;
 *  GraphVertex v1 = g.add_vertex (source);
 *  GraphVertex v2 = g.add_vertex (target);
 *  g.add_edge (v1, v2);
 *
 * Issues
 * Target does not belong here.
 */
class Dag: public VolatilePtr<Dag>
{
public:
    Dag();

    const Graph& g() const { return g_; }

    std::list<Target::Ptr> target;
    // list targets (targets should have a timestamp so that the scheduler can know what to focus on first)
    // this list is publicly accesible

    // invalidate steps (only deprecateCache(Interval::Interval_ALL) for now)
    void deprecateCache(GraphVertex);

    GraphVertex getVertex(Step::Ptr s) const;

    GraphVertex appendStep(Step::Ptr step, GraphVertex gv=boost::graph_traits<Graph>::null_vertex ());
    void removeStep(Step::Ptr step);
    std::vector<Step::Ptr> sourceSteps(Step::Ptr step) const;
    std::vector<Step::Ptr> targetSteps(Step::Ptr step) const;

    static void test();

private:
    Graph g_;

    typedef std::map<Step::Ptr, GraphVertex> StepVertexMap;
    StepVertexMap map;
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_DAG_H
