#ifndef SIGNAL_PROCESSING_DAG_H
#define SIGNAL_PROCESSING_DAG_H

#include "step.h"

#include "volatileptr.h"

#include <boost/graph/directed_graph.hpp>

namespace Signal {
namespace Processing {

typedef boost::directed_graph<Signal::Processing::Step::Ptr, boost::vecS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor GraphVertex;
typedef boost::graph_traits<Graph>::edge_descriptor GraphEdge;
GraphVertex NullVertex();


/**
 * @brief The Dag class should manage the connections between the steps in the signal processing chain.
 *
 * TODO It should treat Step's that aren't a part of the Dag as lonely islands.
 */
class Dag: public VolatilePtr<Dag>
{
public:
    Dag();

    const Graph& g() const { return g_; }


    GraphVertex getVertex(Step::Ptr s) const;

    GraphVertex appendStep(Step::Ptr step, GraphVertex gv=NullVertex ());
    GraphVertex insertStep(Step::Ptr step, GraphVertex gv=NullVertex ());
    void removeStep(Step::Ptr step);
    std::vector<Step::Ptr> sourceSteps(Step::Ptr step) const;
    std::vector<Step::Ptr> targetSteps(Step::Ptr step) const;

private:
    Graph g_;

    typedef std::map<Step::Ptr, GraphVertex> StepVertexMap;
    StepVertexMap map;

public:
    static void test();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_DAG_H
