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

    // Add worker threads to occupy all kernels (the engine above occupies the first)
    for (int i=0; i<QThread::idealThreadCount ()-1; i++) {
        write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
    }

    Chain::Ptr chain(new Chain(dag, targets, workers, bedroom));

    return chain;
}


Chain::
        ~Chain()
{
    // Remove all edges, all vertices and their properties (i.e Step::Ptr)
    dag_.reset ();

    print_dead_workers();

    // Workers will ask all instances of Worker to quit at will. class Workers
    // does not know about any sleeping. class Bedroom takes care of waking
    // them up so that the Worker threads can quit.
    workers_.reset ();

    // Notify sleeping workers that something has changed. They will notice
    // that there's nothing to work on anymore and close.
    bedroom_->wakeup();
}


TargetNeeds::Ptr Chain::
        addTarget(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at)
{
    Step::Ptr step = insertStep(Dag::WritePtr(dag_), desc, at);

    TargetNeeds::Ptr target_needs = write1(targets_)->addTarget(step);

    return target_needs;
}


IInvalidator::Ptr Chain::
        addOperationAt(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = insertStep(Dag::WritePtr(dag_), desc, at);

    IInvalidator::Ptr graph_invalidator( new GraphInvalidator(dag_, bedroom_, step));

    return graph_invalidator;
}


void Chain::
        removeOperationsAt(TargetNeeds::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = read1(at)->step().lock();
    if (!step)
        return;

    Dag::WritePtr dag(dag_);

    GraphVertex v = dag->getVertex (step);
    const Graph& g = dag->g ();

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
    Step::Ptr step = read1(at)->step().lock();
    if (!step)
        return Signal::Interval();

    Dag::ReadPtr dag(dag_);

    Graph rev; ReverseGraph::reverse_graph (dag->g (), rev);
    GraphVertex at_vertex = ReverseGraph::find_first_vertex (rev, step);

    boost::optional<Signal::Interval> I;
    breadth_first_search(rev, at_vertex, visitor(find_extent(&I)));

    return I.get_value_or (Signal::Interval());
}


void Chain::
        print_dead_workers() const
{
    Workers::DeadEngines engines = write1(workers_)->clean_dead_workers();
    Workers::print (engines);
}


void Chain::
        rethrow_worker_exception() const
{
    write1(workers_)->rethrow_worker_exception();
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
    if (at) {
        Step::Ptr target_step = read1(at)->step().lock();
        EXCEPTION_ASSERTX (target_step, "target step has been removed");

        vertex = dag->getVertex (target_step);
    }

    Step::Ptr step(new Step(desc));
    dag->appendStep (step, vertex);

    return step;
}


class OperationDescChainMock : public Signal::OperationDesc
{
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* J) const {
        if (J) *J = I;
        return I;
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
        IInvalidator::Ptr invalidator = write1(chain)->addOperationAt(source_desc, target);

        EXCEPTION_ASSERT_EQUALS (read1(chain)->extent(target), Signal::Interval(3,5));

        write1(target)->updateNeeds(Signal::Interval(4,6));
        usleep(4000);
        //target->sleep();

        // This will remove the step used by invalidator
        write1(chain)->removeOperationsAt(target);

        // So using invalidator should not do anything (would throw an
        // exception if OperationDescChainMock::affectedInterval was called)
        write1(invalidator)->deprecateCache(Signal::Interval(9,11));

        usleep(4000);
        read1(chain)->rethrow_worker_exception();
    }
    // Wait for threads to exit.
    usleep(4000);
}

} // namespace Processing
} // namespace Signal
