#include "chain.h"
#include "bedroom.h"
#include "firstmissalgorithm.h"
#include "targetschedule.h"
#include "reversegraph.h"
#include "graphinvalidator.h"
#include "bedroomnotifier.h"
#include "workers.h"

#include "timer.h"
#include "tasktimer.h"

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
    BedroomNotifier::Ptr notifier(new BedroomNotifier(bedroom));
    Targets::Ptr targets(new Targets(notifier));

    IScheduleAlgorithm::Ptr algorithm(new FirstMissAlgorithm());
    ISchedule::Ptr targetSchedule(new TargetSchedule(dag, algorithm, targets));
    Workers::Ptr workers(new Workers(targetSchedule, bedroom));

    // Add the 'single instance engine' thread.
    workers.write ()->addComputingEngine(Signal::ComputingEngine::Ptr());

    // Add worker threads to occupy all kernels
    for (int i=0; i<QThread::idealThreadCount (); i++) {
        workers.write ()->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
    }

    Chain::Ptr chain(new Chain(dag, targets, workers, bedroom, notifier));

    return chain;
}


Chain::
        ~Chain()
{
    close();
}


bool Chain::
        close(int timeout)
{
    if (!workers_)
        return true;

    TaskTimer tt("Chain::close(%d)", timeout);
    // Targets::TargetNeedsCollection T = targets_.read ()->getTargets();

    // Ask workers to not start anything new
    workers_.read ()->remove_all_engines(0);

    // Make scheduler return to worker
    bedroom_->close();

    // Wait 1.0 s for workers to finish
    bool r = workers_.read ()->remove_all_engines(timeout);

    // Suppress output
    workers_.write ()->clean_dead_workers();

    // Remove all workers
    workers_ = shared_state<Workers> ();

    // Remove all edges, all vertices and their properties (i.e Step::Ptr)
    dag_ = Dag::Ptr ();

    targets_ = Targets::Ptr ();
    bedroom_.reset ();
    notifier_.reset ();

    return r;
}


TargetMarker::Ptr Chain::
        addTarget(Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    Step::Ptr::weak_ptr step = createBranchStep(*dag_.write (), desc, at);

    TargetNeeds::Ptr target_needs = targets_.write ()->addTarget(step);

    TargetMarker::Ptr marker(new TargetMarker {target_needs, dag_});

    return marker;
}


IInvalidator::Ptr Chain::
        addOperationAt(Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr::weak_ptr step = insertStep(*dag_.get (), desc, at);

    IInvalidator::Ptr graph_invalidator( new GraphInvalidator {dag_, notifier_, step});

    desc.write ()->setInvalidator( graph_invalidator );

    graph_invalidator.read ()->deprecateCache (Signal::Interval::Interval_ALL);

    return graph_invalidator;
}


void Chain::
        removeOperationsAt(TargetMarker::Ptr at)
{
    EXCEPTION_ASSERT (at);

    Step::Ptr step = at->step().lock();
    if (!step)
        return;

    auto dag = dag_.write ();

    GraphVertex v = dag->getVertex (step);
    if (!v)
        return;

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
        Step::Ptr step( g[u] ); // lock while studying what's needed
        Signal::OperationDesc::Extent x = step->operation_desc()->extent ();

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
    Signal::OperationDesc::Extent E;

    Step::Ptr step = at->step().lock();
    if (!step)
        return E;

    Graph rev; ReverseGraph::reverse_graph (dag_.read ()->g (), rev);
    GraphVertex at_vertex = ReverseGraph::find_first_vertex (rev, step);

    if (at_vertex)
        breadth_first_search(rev, at_vertex, visitor(find_extent(&E)));

    return E;
}


Workers::Ptr Chain::
        workers() const
{
    return workers_;
}


Targets::Ptr Chain::
        targets() const
{
    return targets_;
}


Chain::
        Chain(Dag::Ptr dag, Targets::Ptr targets, Workers::Ptr workers, Bedroom::Ptr bedroom, INotifier::Ptr notifier)
    :
      dag_(dag),
      targets_(targets),
      workers_(workers),
      bedroom_(bedroom),
      notifier_(notifier)
{
}


Step::Ptr::weak_ptr Chain::
        createBranchStep(Dag& dag, Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    GraphVertex vertex = NullVertex ();
    if (at) {
        Step::Ptr target_step = at->step().lock();
        EXCEPTION_ASSERTX (target_step, "target step has been removed");

        vertex = dag.getVertex (target_step);
        if (!vertex)
            return Step::Ptr::weak_ptr();

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


Step::Ptr::weak_ptr Chain::
        insertStep(Dag& dag, Signal::OperationDesc::Ptr desc, TargetMarker::Ptr at)
{
    GraphVertex vertex = NullVertex ();
    if (at) {
        Step::Ptr target_step = at->step().lock();
        EXCEPTION_ASSERTX (target_step, "target step has been removed");

        vertex = dag.getVertex (target_step);
        if (!vertex)
            return Step::Ptr::weak_ptr();
    }

    Step::Ptr step(new Step(desc));
    dag.insertStep (step, vertex);

    return step;
}


} // namespace Processing
} // namespace Signal

#include "test/operationmockups.h"
#include "test/randombuffer.h"
#include "signal/buffersource.h"
#include <QApplication>

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
    std::string name = "Chain";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);

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

    {
        Timer t;
        Chain::createDefaultChain ();
        EXCEPTION_ASSERT_LESS (t.elapsed (), 0.01);
    }

    // It should make the signal processing namespace easy to use with a clear
    // and simple interface.
    {
        Timer t;
        Chain::Ptr chain = Chain::createDefaultChain ();
        Signal::OperationDesc::Ptr target_desc(new OperationDescChainMock);
        Signal::OperationDesc::Ptr source_desc(new Signal::BufferSource(Test::RandomBuffer::smallBuffer ()));

        TargetMarker::Ptr null;
        TargetMarker::Ptr target = chain.write ()->addTarget(target_desc, null);

        // Should be able to add and remove an operation multiple times
        chain.write ()->addOperationAt(source_desc, target);
        chain.write ()->removeOperationsAt(target);
        chain.write ()->addOperationAt(source_desc, target);
        chain.write ()->extent(target); // will fail unless indices are reordered
        EXCEPTION_ASSERT_EQUALS (chain->dag_.read ()->g().num_edges(), 1u);
        EXCEPTION_ASSERT_EQUALS (chain->dag_.read ()->g().num_vertices(), 2u);
        chain.write ()->removeOperationsAt(target);


        // Should create an invalidator when adding an operation
        IInvalidator::Ptr invalidator = chain.write ()->addOperationAt(source_desc, target);

        EXCEPTION_ASSERT_EQUALS (chain.read ()->extent(target).interval, Signal::Interval(3,5));

        TargetNeeds::Ptr needs = target->target_needs();
        needs.write ()->updateNeeds(Signal::Interval(4,6));
        usleep(4000);
        //target->sleep();

        // This will remove the step used by invalidator
        chain.write ()->removeOperationsAt(target);

        // So using invalidator should not do anything (would throw an
        // exception if OperationDescChainMock::affectedInterval was called)
        invalidator.write ()->deprecateCache(Signal::Interval(9,11));

        usleep(4000);
        chain.read ()->workers()->rethrow_any_worker_exception();

        chain = Chain::Ptr ();
        EXCEPTION_ASSERT_LESS(t.elapsed (), 0.03);
    }
}

} // namespace Processing
} // namespace Signal
