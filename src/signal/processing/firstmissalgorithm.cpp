#include "firstmissalgorithm.h"

#include "reversegraph.h"

#include <boost/foreach.hpp>
#include <boost/graph/breadth_first_search.hpp>

//#define DEBUGINFO
#define DEBUGINFO if(0)

using namespace boost;

namespace Signal {
namespace Processing {


typedef std::map<GraphVertex, Signal::Intervals> NeededSamples;


struct ScheduleParams {
    Signal::ComputingEngine::Ptr engine;
    Signal::IntervalType preferred_size;
    Signal::IntervalType center;
};


class find_missing_samples: public default_bfs_visitor {
public:
    find_missing_samples(NeededSamples needed, Task::Ptr* output_task, ScheduleParams schedule_params)
        :
          needed(needed),
          params(schedule_params),
          task(output_task)
    {
    }


    void discover_vertex(GraphVertex u, const Graph & g)
    {
        if (*task)
            return;

        DEBUGINFO TaskTimer tt(format("discover_vertex %1%") % u);

        Step::WritePtr step( g[u] ); // lock while studying what's needed
        Signal::Intervals I = needed[u] & step->not_started ();
        Signal::OperationDesc::Ptr od = step->operation_desc();

        // Compute what we need from sources
        DEBUGINFO TaskInfo(format("step %1%: missing samples %2%") % ((void*)&*step) % I);
        if (!I)
            return;

        Signal::Interval expected_output = I.fetchInterval(params.preferred_size, params.center);
        Signal::Intervals required_input;
        for (Signal::Intervals x = expected_output; x;) {
            Signal::Interval actual_output;
            Signal::Interval r1 = od->requiredInterval (x, &actual_output);
            required_input |= r1;
            EXCEPTION_ASSERTX (actual_output & x,
                               boost::format("actual_output = %1%, x = %2%")
                               % actual_output % x); // check for valid 'requiredInterval' by making sure that actual_output doesn't stall needed
            x -= actual_output;
        }

        // Compute what the sources have available
        Intervals total_missing;
        BOOST_FOREACH(GraphEdge e, out_edges(u, g)) {
            GraphVertex v = target(e,g);
            Step::ReadPtr src( g[v] );
            needed[v] |= src->not_started () & required_input;
            total_missing |= src->out_of_date () & required_input;
        }

        // If there are no sources
        if (0==out_degree(u, g)) {
            // Then this operation must specify sample rate and number of
            // samples for this to be a valid read. Otherwise the signal is
            // undefined.
            Signal ::OperationDesc::Extent x = od->extent ();
            if (!x.number_of_channels.is_initialized () || !x.sample_rate.is_initialized ())
                total_missing = Signal::Interval::Interval_ALL; // A non-empty interval
        }

        // If nothing is missing and this engine supports this operation
        if (total_missing.empty () && step->operation (params.engine))
            // Even if this engine doesn't support this operation it should
            // still update 'needed' so that it can compute what's
            // needed in the children.
        {
            // Create a task
            std::vector<Step::Ptr> children;
            BOOST_FOREACH(GraphEdge e, out_edges(u, g)) {
                GraphVertex v = target(e,g);
                children.push_back (g[v]);
            }

            task->reset (new Task(&*step, g[u], children, expected_output));
        }
    }

    NeededSamples needed;
    ScheduleParams params;
    Task::Ptr* task;
};


Task::Ptr FirstMissAlgorithm::
        getTask(const Graph& straight_g,
                GraphVertex straight_target,
                Signal::Intervals needed,
                Signal::IntervalType center,
                Workers::Ptr workers,
                Signal::ComputingEngine::Ptr engine) const
{
    DEBUGINFO TaskTimer tt("getTask");
    Graph g; ReverseGraph::reverse_graph (straight_g, g);
    GraphVertex target = ReverseGraph::find_first_vertex (g, straight_g[straight_target]);

    int preferred_size = needed.count ();

    if (workers)
        preferred_size = 1 + preferred_size/read1(workers)->n_workers();

    ScheduleParams schedule_params = { engine, preferred_size, center };

    NeededSamples needed_samples;
    needed_samples[target] = needed;


    Task::Ptr task;
    find_missing_samples vis(needed_samples, &task, schedule_params);

    breadth_first_search(g, target, visitor(vis));

    if (!task)
        DEBUGINFO TaskInfo("didn't find anything");

    return task;
}


void FirstMissAlgorithm::
        test()
{
    // It should figure out the missing pieces in the graph and produce a Task to work it off
    {
        // Create an OperationDesc and a Step
        Signal::pBuffer b(new Buffer(Interval(60,70), 40, 7));
        Signal::OperationDesc::Ptr od(new BufferSource(b));
        Step::Ptr step(new Step(od));

        // Create a graph with only one vertex
        Graph g;
        GraphVertex v = g.add_vertex (step);


        // Schedule a task
        FirstMissAlgorithm schedule;
        Task::Ptr t1 = schedule.getTask(g, v, Signal::Interval(20,30), 25);
        Task::Ptr t2 = schedule.getTask(g, v, Signal::Interval(10,24) | Signal::Interval(26,30), 25);


        // Verify output
        EXCEPTION_ASSERT(t1);
        EXCEPTION_ASSERT(t2);
        EXCEPTION_ASSERT_EQUALS(read1(t1)->expected_output(), Interval(20,30));
        EXCEPTION_ASSERT_EQUALS(read1(t2)->expected_output(), Interval(10, 20));

        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), Signal::Intervals::Intervals_ALL);
        EXCEPTION_ASSERT_EQUALS(~Signal::Intervals(10,30), read1(step)->not_started());

        // Verify that the output objects can be used
        t1->run(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
        t2->run(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), read1(step)->not_started());
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), ~Signal::Intervals(10,30));
    }

    // It should let missing_in_target override out_of_date in the given vertex
}


} // namespace Processing
} // namespace Signal
