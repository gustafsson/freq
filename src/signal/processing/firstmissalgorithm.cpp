#include "firstmissalgorithm.h"

#include "reversegraph.h"

#include <boost/foreach.hpp>
#include <boost/graph/breadth_first_search.hpp>

using namespace boost;

namespace Signal {
namespace Processing {


typedef std::map<GraphVertex, Signal::Intervals> MissingSamples;


struct ScheduleParams {
    Signal::IntervalType preferred_size;
    Signal::IntervalType center;
};


class find_missing_samples: public default_bfs_visitor {
public:
    find_missing_samples(MissingSamples missing_samples, Task::Ptr* output_task, ScheduleParams schedule_params)
        :
          missing_samples(missing_samples),
          params(schedule_params),
          task(output_task)
    {
    }


    void discover_vertex(GraphVertex u, const Graph & g)
    {
        if (*task)
            return;

        Step::WritePtr step( g[u] ); // lock while studying what's needed
        Signal::Intervals I = missing_samples[u];
        Signal::OperationDesc::Ptr od = step->operation_desc();

        // Compute what we need from sources
        Signal::Interval expected_output = I.fetchInterval(params.preferred_size, params.center);
        Signal::Intervals required_input;
        for (Signal::Intervals needed = expected_output; needed;) {
            Signal::Interval actual_output;
            Signal::Interval r1 = od->requiredInterval (needed, &actual_output);
            required_input |= r1;
            EXCEPTION_ASSERTX (actual_output & needed,
                               boost::format("actual_output = %1%, needed = %2%")
                               % actual_output % needed); // check for valid 'requiredInterval' by making sure that actual_output doesn't stall needed
            needed -= actual_output;
        }

        // Compute what the sources have available
        Intervals total_missing;
        BOOST_FOREACH(GraphEdge e, out_edges(u, g)) {
            GraphVertex v = target(e,g);
            Step::ReadPtr src( g[v] );
            Signal::Intervals src_missing = src->out_of_date() & required_input;
            missing_samples[v] |= src_missing;
            total_missing |= src_missing;
        }

        // If nothing is missing
        if (total_missing.empty ())
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

    MissingSamples missing_samples;
    ScheduleParams params;
    Task::Ptr* task;
};


Task::Ptr FirstMissAlgorithm::
        getTask(const Graph& straight_g,
                GraphVertex straight_target,
                Signal::Intervals missing_in_target,
                Signal::IntervalType center,
                Workers::Ptr workers,
                Signal::ComputingEngine::Ptr /*engine*/) const
{
    Graph g; ReverseGraph::reverse_graph (straight_g, g);
    GraphVertex target = ReverseGraph::find_first_vertex (g, straight_g[straight_target]);

    int preferred_size = missing_in_target.count ();

    if (workers)
        preferred_size = 1 + preferred_size/read1(workers)->n_workers();

    ScheduleParams schedule_params = { preferred_size, center };

    MissingSamples missing_samples;
    missing_samples[target] = missing_in_target;


    Task::Ptr task;
    find_missing_samples vis(missing_samples, &task, schedule_params);

    breadth_first_search(g, target, visitor(vis));

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
        Task::Ptr t2 = schedule.getTask(g, v, Signal::Interval(20,24) | Signal::Interval(26,30), 25);


        // Verify output
        EXCEPTION_ASSERT_EQUALS(read1(t1)->expected_output(), Interval(20,30));
        EXCEPTION_ASSERT_EQUALS(read1(t2)->expected_output(), Interval(20,24));

        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), Signal::Intervals::Intervals_ALL);
        EXCEPTION_ASSERT_EQUALS(~Signal::Intervals(20,30), read1(step)->not_started());

        // Verify that the output objects can be used
        t1->run(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
        t2->run(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), read1(step)->not_started());
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), ~Signal::Intervals(20,30));
    }

    // It should let missing_in_target override out_of_date in the given vertex
}


} // namespace Processing
} // namespace Signal
