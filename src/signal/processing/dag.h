#ifndef SIGNAL_PROCESSING_DAG_H
#define SIGNAL_PROCESSING_DAG_H

#include <boost/graph/directed_graph.hpp>

#include "step.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GraphProperties class describes the signal processing chain
 *
 * example:
 *
 *  Graph g;
 *  Signal::Processing::Step::Ptr source, target;
 *  GraphVertex v1 = g.add_vertex (source);
 *  GraphVertex v2 = g.add_vertex (target);
 *  g.add_edge (v1, v2);
 */
class GraphProperties
{
public:
    GraphProperties();

    static void test();
};


typedef boost::directed_graph<Signal::Processing::Step::Ptr, boost::vecS, GraphProperties> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor GraphVertex;
typedef boost::graph_traits<Graph>::edge_descriptor GraphEdge;

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_DAG_H
