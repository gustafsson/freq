#ifndef SIGNAL_PROCESSING_REVERSEGRAPH_H
#define SIGNAL_PROCESSING_REVERSEGRAPH_H

#include "dag.h"

namespace Signal {
namespace Processing {

class ReverseGraph
{
public:
    /**
     * @brief reverse_graph writes a reverse copy of the directional graph g into h.
     * @param g
     * @param h
     */
    static void reverse_graph(const Graph& g, Graph& h);


    /**
     * @brief find_first_vertex returns the first vertex in g which property equals p.
     * @param g
     * @param p
     * @return graph_traits<Graph>::null_vertex () if p was not found in g.
     */
    static GraphVertex find_first_vertex(const Graph& g, Step::ptr p);
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_REVERSEGRAPH_H
