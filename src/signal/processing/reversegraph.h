#ifndef SIGNAL_PROCESSING_REVERSEGRAPH_H
#define SIGNAL_PROCESSING_REVERSEGRAPH_H

#include "dag.h"

namespace Signal {
namespace Processing {

class ReverseGraph
{
public:
    static void reverse_graph(const Graph& g, Graph&);

    static GraphVertex find_first_vertex(const Graph& g, Step::Ptr property);
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_REVERSEGRAPH_H
