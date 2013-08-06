#include "chain.h"
#include "bedroom.h"
#include "firstmissalgorithm.h"
#include "sleepschedule.h"
#include "targetschedule.h"
#include "reversegraph.h"
#include "graphinvalidator.h"

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
        //write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
    }

    Chain::Ptr chain(new Chain(dag, targets, workers, bedroom));

    return chain;
}


Chain::
        ~Chain()
{
    print_dead_workers();

    // Workers will ask all instances of Worker to quit at will. class Workers
    // does not know about any sleeping. class Bedroom takes care of waking
    // them up so that the Worker threads can quit.
    workers_.reset ();

    // Notify sleeping workers that something has changed. They will notice
    // that there's nothing to work on anymore and close.
    bedroom_->wakeup();

    // Wait for targets to finish.
    // 1.0 s 'should' be enough.
    // If it isn't, well then there will probably be a crash if
    // OpenGL resources are released from a different thread.
    Targets::TargetNeedsCollection T = read1(targets_)->getTargets();

    BOOST_FOREACH( TargetNeeds::Ptr t, T ) {
        write1(t)->updateNeeds(
                    Signal::Intervals(),
                    0,
                    Signal::Interval::IntervalType_MIN,
                    Signal::Intervals());

        bedroom_->wakeup();
        t->sleep(1000);
    }

    // Remove all edges, all vertices and their properties (i.e Step::Ptr)
    dag_.reset ();
}


TargetMarker::Ptr Chain::
        addTarget(Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    Step::Ptr step = createBranchStep(*Dag::WritePtr(dag_), desc, at);

    TargetNeeds::Ptr target_needs = write1(targets_)->addTarget(step);

    TargetMarker::Ptr marker(new TargetMarker(target_needs, dag_));

    return marker;
}


IInvalidator::Ptr Chain::
        addOperationAt(Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = insertStep(*Dag::WritePtr(dag_), desc, at);

    IInvalidator::Ptr graph_invalidator( new GraphInvalidator(dag_, bedroom_, step));

    read1(graph_invalidator)->deprecateCache(Signal::Interval::Interval_ALL);

    return graph_invalidator;
}


void Chain::
        removeOperationsAt(TargetMarker::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = read1(at)->step().lock();
    if (!step)
        return;

    Dag::WritePtr dag(dag_);

    GraphVertex v = dag->getVertex (step);
    const Graph& g = dag->g ();

    std::vector<Step::Ptr> steps_to_remove;
    BOOST_FOREACH(GraphEdge e, in_edges(v, g)) {
        Step::Ptr s = g[source(e, g)];
        steps_to_remove.push_back (s);
    }

    BOOST_FOREACH(Step::Ptr s, steps_to_remove) {
        GraphInvalidator::deprecateCache (*dag, s, Signal::Interval::Interval_ALL);
        dag->removeStep (s);
    }

    bedroom_->wakeup();
}


class find_extent: public default_bfs_visitor {
public:
    find_extent(Signal::OperationDesc::Extent* extent)
        :   extent(extent)
    {
    }


    void discover_vertex(GraphVertex u, const Graph & g)
    {
        Step::WritePtr step( g[u] ); // lock while studying what's needed
        Signal::OperationDesc::Ptr od = step->operation_desc();
        Signal::OperationDesc::Extent x = od->extent ();

        // TODO This doesn't really work with merged paths
        // But it could be extended to support that by merging the extents of merged paths.

        if (!extent->interval.is_initialized ())
            extent->interval = x.interval;

        if (!extent->number_of_channels.is_initialized ())
            extent->number_of_channels = x.number_of_channels;

        if (!extent->sample_rate.is_initialized ())
            extent->sample_rate = x.sample_rate;
    }

    Signal::OperationDesc::Extent* extent;
};


Signal::OperationDesc::Extent Chain::
        extent(TargetMarker::Ptr at) const
{
    Step::Ptr step = read1(at)->step().lock();
    if (!step)
        return Signal::OperationDesc::Extent();

    Dag::ReadPtr dag(dag_);

    Graph rev; ReverseGraph::reverse_graph (dag->g (), rev);
    GraphVertex at_vertex = ReverseGraph::find_first_vertex (rev, step);

    Signal::OperationDesc::Extent I;
    breadth_first_search(rev, at_vertex, visitor(find_extent(&I)));

    return I;
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
        createBranchStep(Dag& dag, Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    GraphVertex vertex = graph_traits<Graph>::null_vertex ();
    if (at) {
        Step::Ptr target_step = read1(at)->step().lock();
        EXCEPTION_ASSERTX (target_step, "target step has been removed");

        vertex = dag.getVertex (target_step);
        BOOST_FOREACH(const GraphEdge& e, in_edges(vertex, dag.g ())) {
            // Pick one of the sources on random and append to that one
            vertex = source(e, dag.g ());
            break;
        }
    }

    Step::Ptr step(new Step(desc));
    dag.appendStep (step, vertex);

    return step;
}


Step::Ptr Chain::
        insertStep(Dag& dag, Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    GraphVertex vertex = graph_traits<Graph>::null_vertex ();
    if (at) {
        Step::Ptr target_step = read1(at)->step().lock();
        EXCEPTION_ASSERTX (target_step, "target step has been removed");

        vertex = dag.getVertex (target_step);
    }

    Step::Ptr step(new Step(desc));
    dag.insertStep (step, vertex);

    return step;
}


} // namespace Processing
} // namespace Signal

#include "test/operationmockups.h"

namespace Signal {
namespace Processing {

class OperationDescChainMock : public Test::TransparentOperationDesc
{
    Extent extent() const {
        Extent x;
        x.interval = Signal::Interval(3,5);
        return x;
    }
};


void Chain::
        test()
{
    // Boost graph shall support removing and adding vertices without breaking color maps
    {
        typedef directed_graph<> my_graph;
        typedef graph_traits<my_graph>::vertex_descriptor my_vertex;

        my_graph g;
        my_vertex v1 = g.add_vertex ();
        g.remove_vertex (v1);
        g.renumber_indices (); // required after removing a vertex
        my_vertex v2 = g.add_vertex ();
        breadth_first_search(g, v2, visitor(default_bfs_visitor()));
    }


    // It should make the signal processing namespace easy to use with a clear
    // and simple interface.
    {
        Chain::Ptr chain = Chain::createDefaultChain ();
        Signal::OperationDesc::Ptr target_desc(new OperationDescChainMock);
        Signal::OperationDesc::Ptr source_desc(new OperationDescChainMock);

        TargetMarker::Ptr null;
        TargetMarker::Ptr target = write1(chain)->addTarget(target_desc, null);

        // Should be able to add and remove an operation multiple times
        write1(chain)->addOperationAt(source_desc, target);
        write1(chain)->removeOperationsAt(target);
        write1(chain)->addOperationAt(source_desc, target);
        write1(chain)->extent(target); // will fail unless indices are reordered
        EXCEPTION_ASSERT_EQUALS (read1(chain->dag_)->g().num_edges(), 1);
        EXCEPTION_ASSERT_EQUALS (read1(chain->dag_)->g().num_vertices(), 2);
        write1(chain)->removeOperationsAt(target);


        // Should create an invalidator when adding an operation
        IInvalidator::Ptr invalidator = write1(chain)->addOperationAt(source_desc, target);

        EXCEPTION_ASSERT_EQUALS (read1(chain)->extent(target).interval, Signal::Interval(3,5));

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
