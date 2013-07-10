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
 * @brief The Dag class describes the signal processing chain
 *
 * example:
 *
 *  Graph g;
 *  Signal::Processing::Step::Ptr source, target;
 *  GraphVertex v1 = g.add_vertex (source);
 *  GraphVertex v2 = g.add_vertex (target);
 *  g.add_edge (v1, v2);
 */
class Dag: public VolatilePtr<Dag>
{
public:
    Dag();

    Graph g;

    std::list<Target::Ptr> target;
    // list targets (targets should have a timestamp so that the scheduler can know what to focus on first)

    // invalidate steps (only deprecateCache(Interval::Interval_ALL) for now)
    void deprecateCache(GraphVertex);

    // map step to vertex
    std::map<Step::Ptr, GraphVertex> map;

    // manage add/remove vertex from graph, throw exception if not found
    void insertStep(GraphVertex gv, Step::Ptr step);
    void removeStep(Step::Ptr step);

    static void test();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_DAG_H
