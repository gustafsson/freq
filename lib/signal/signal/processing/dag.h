#ifndef SIGNAL_PROCESSING_DAG_H
#define SIGNAL_PROCESSING_DAG_H

#include "step.h"

#include "shared_state.h"

#include <boost/graph/directed_graph.hpp>

namespace Signal {
namespace Processing {

typedef boost::directed_graph<Signal::Processing::Step::ptr, boost::vecS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor GraphVertex;
typedef boost::graph_traits<Graph>::edge_descriptor GraphEdge;
GraphVertex NullVertex();


/**
 * @brief The Dag class should manage the connections between the steps in the signal processing chain.
 *
 * It should treat Step's that aren't a part of the Dag as lonely islands.
 */
class Dag
{
public:
    typedef shared_state<Dag> ptr;

    Dag();

    const Graph& g() const { return g_; }

    /**
     * @brief getVertex locates where 's' is in the Dag.
     * @param s The property to search for.
     * @return The GraphVertex with 's' as a property (there's shoule only be
     * one). If there is no such vertex NullVertex is returned. This may happen
     * if the Step is removed from the Dag in another thread  while a Step::Ptr
     * is kept in this thread.
     * Note that NullVertex converts to false in a boolean expression.
     * Note that most graph functions will crash if you give them a NullVertex.
     */
    GraphVertex getVertex(Step::ptr s) const;

    GraphVertex appendStep(Step::ptr step, GraphVertex gv=NullVertex ());
    GraphVertex insertStep(Step::ptr step, GraphVertex gv=NullVertex ());
    void removeStep(Step::ptr step);
    void removeOperation(OperationDesc::ptr op);
    std::vector<Step::ptr> sourceSteps(Step::ptr step) const;
    std::vector<Step::ptr> targetSteps(Step::ptr step) const;

private:
    Graph g_;

    typedef std::map<Step::ptr, GraphVertex> StepVertexMap;
    StepVertexMap map;

public:
    static void test();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_DAG_H
