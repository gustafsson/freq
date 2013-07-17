#include "chain.h"
#include "bedroom.h"
#include "firstmissalgorithm.h"
#include "sleepschedule.h"
#include "targetschedule.h"
#include "reversegraph.h"

#include <boost/foreach.hpp>
#include <boost/graph/breadth_first_search.hpp>

using namespace boost;

namespace Signal {
namespace Processing {


Chain::Ptr Chain::
        createDefaultChain()
{
    Dag::Ptr dag(new Dag);
    Bedroom::Ptr bedroom(new Bedroom);
    Targets::Ptr targets(new Targets(bedroom));

    IScheduleAlgorithm::Ptr algorithm(new FirstMissAlgorithm());
    ISchedule::Ptr targetSchedule(new TargetSchedule(dag, algorithm, targets));
    ISchedule::Ptr sleepSchedule(new SleepSchedule(bedroom, targetSchedule));
    Workers::Ptr workers(new Workers(sleepSchedule));

    // Add the 'single instance engine' thread.
    write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr());

//    for (int i=0; i<QThread::idealThreadCount (); i++) {
//        write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
//    }

    Chain::Ptr chain(new Chain(dag, targets, workers, bedroom));

    return chain;
}


TargetNeeds::Ptr Chain::
        addTarget(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at)
{
    Step::Ptr step = insertStep(Dag::WritePtr(dag_), desc, at);

    TargetNeeds::Ptr target_needs = write1(targets_)->addTarget(step);

    return target_needs;
}


IInvalidator::Ptr Chain::
        addOperation(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = insertStep(Dag::WritePtr(dag_), desc, at);

    IInvalidator::Ptr graph_invalidator( new GraphInvalidator(dag_, bedroom_, step));

    return graph_invalidator;
}


void Chain::
        removeOperations(TargetNeeds::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Dag::WritePtr dag(dag_);

    const Graph& g = dag->g ();
    GraphVertex v = dag->getVertex (read1(at)->step());

    std::vector<GraphEdge> inedges;
    BOOST_FOREACH(GraphEdge e, in_edges(v, g)) {
        inedges.push_back (e);
    }

    BOOST_FOREACH(GraphEdge e, inedges) {
        dag->removeStep (g[source(e, g)]);
    }
}


class find_extent: public default_bfs_visitor {
public:
    find_extent(boost::optional<Signal::Interval>* extent)
        :   extent(extent)
    {
    }


    void discover_vertex(GraphVertex u, const Graph & g)
    {
        if (*extent)
            return;

        Step::WritePtr step( g[u] ); // lock while studying what's needed
        Signal::OperationDesc::Ptr od = step->operation_desc();
        Signal::OperationDesc::Extent x = od->extent ();

        // This doesn't really work with merged paths
        // But it could be extended to support that by merging the extents of merged paths.
        *extent = x.interval;
    }

    boost::optional<Signal::Interval>* extent;
};


Signal::Interval Chain::
        extent(TargetNeeds::Ptr at) const
{
    Dag::ReadPtr dag(dag_);
    Step::Ptr step = read1(at)->step();

    Graph rev; ReverseGraph::reverse_graph (dag->g (), rev);
    GraphVertex at_vertex = ReverseGraph::find_first_vertex (rev, step);

    boost::optional<Signal::Interval> I;
    breadth_first_search(rev, at_vertex, visitor(find_extent(&I)));

    return I.get_value_or (Signal::Interval());
}


Chain::
        Chain(Dag::Ptr dag, Targets::Ptr targets, Workers::Ptr workers, Bedroom::Ptr bedroom)
    :
      dag_(dag),
      targets_(targets),
      workers_(workers),
      bedroom_(bedroom)
{
}


Step::Ptr Chain::
        insertStep(const Dag::WritePtr& dag, Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at)
{
    GraphVertex vertex = boost::graph_traits<Graph>::null_vertex ();
    if (at)
        vertex = dag->getVertex (read1(at)->step());

    Step::Ptr step(new Step(desc));
    dag->appendStep (step, vertex);

    return step;
}


class OperationDescChainMock : public Signal::OperationDesc
{
    Signal::Interval requiredInterval( const Signal::Interval&, Signal::Interval* ) const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return Signal::Interval();
    }

    Signal::Interval affectedInterval( const Signal::Interval& ) const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return Signal::Interval();
    }

    OperationDesc::Ptr copy() const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return OperationDesc::Ptr();
    }

    Operation::Ptr createOperation( ComputingEngine* ) const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return Operation::Ptr();
    }

    Extent extent() const {
        Extent x;
        x.interval = Signal::Interval(3,5);
        return x;
    }
};


void Chain::
        test()
{
    // It should make the signal processing namespace easy to use with a clear
    // and simple interface.
    {
        Chain::Ptr chain = Chain::createDefaultChain ();

        Signal::OperationDesc::Ptr target_desc(new OperationDescChainMock);
        Signal::OperationDesc::Ptr source_desc(new OperationDescChainMock);

        TargetNeeds::Ptr null;
        TargetNeeds::Ptr target = write1(chain)->addTarget(target_desc, null);
        IInvalidator::Ptr invalidator = write1(chain)->addOperation(source_desc, target);

        EXCEPTION_ASSERT_EQUALS (read1(chain)->extent(target), Signal::Interval(3,5));

        write1(chain)->removeOperations(target);

        write1(target)->updateNeeds(Signal::Interval(4,6));
        write1(invalidator)->deprecateCache(Signal::Interval(9,11));
    }
}

} // namespace Processing
} // namespace Signal
