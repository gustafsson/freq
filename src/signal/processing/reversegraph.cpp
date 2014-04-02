#include "reversegraph.h"

#include <boost/foreach.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/reverse_graph.hpp>

using namespace boost;

namespace Signal {
namespace Processing {


void ReverseGraph::
        reverse_graph(const Graph& g, Graph& h)
{
    copy_graph(make_reverse_graph(g), h);
}


GraphVertex ReverseGraph::
        find_first_vertex(const Graph& g, Step::ptr property)
{
    GraphVertex u = graph_traits<Graph>::null_vertex ();

    BOOST_FOREACH(GraphVertex v, vertices(g)) {
        if (g[v] == property)
            u = v;
    }

    return u;
}


} // namespace Processing
} // namespace Signal
