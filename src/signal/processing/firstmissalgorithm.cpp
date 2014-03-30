#include "firstmissalgorithm.h"

#include "reversegraph.h"
#include "tasktimer.h"
#include "expectexception.h"

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

        // Compute what the sources have available
        Intervals missing_input;
        BOOST_FOREACH(GraphEdge e, out_edges(u, g))
          {
            GraphVertex v = target(e,g);
            Step::ReadPtr src( g[v] );
            missing_input |= src->out_of_date ();
          }

        Interval required_input = try_create_task(u, g, missing_input);

        // Update sources with needed samples
        BOOST_FOREACH(GraphEdge e, out_edges(u, g))
          {
            GraphVertex v = target(e,g);
            Step::ReadPtr src( g[v] );
            needed[v] |= src->not_started () & required_input;
          }
      }


    Signal::Interval try_create_task(GraphVertex u, const Graph & g, Signal::Intervals missing_input)
      {
        Step::WritePtr step( g[u] ); // lock while studying what's needed

        try
          {
            Signal::Intervals I = needed[u] & step->not_started ();

            // Compute what we need from sources
            if (!I)
              {
                // Need to do nothing more here
                return Signal::Interval();
              }

            Signal::OperationDesc::Ptr op = step->operation_desc();
            Signal::OperationDesc::WritePtr o(op);

            DEBUGINFO TaskTimer tt(format("Missing %1% in %2% for %3%")
                                   % I
                                   % o->toString ().toStdString ()
                                   % (params.engine?vartype(*params.engine.get ()):""));


            // params.preferred_size is just a preferred update size, not a required update size.
            // Accept whatever requiredInterval sets as expected_output
            Signal::Interval wanted_output = I.fetchInterval(params.preferred_size, params.center);
            Signal::Interval expected_output;
            Signal::Interval required_input = o->requiredInterval (wanted_output, &expected_output);;
            EXCEPTION_ASSERTX(required_input, o->toString ().toStdString ());

            DEBUGINFO TaskInfo(boost::format("params.preferred_size = %d, params.center = %d, wanted_output = %s")
                     % params.preferred_size % params.center % wanted_output);
            DEBUGINFO TaskInfo(boost::format("expected_output = %s, required_input = %s")
                     % expected_output % required_input);

            // check for valid 'requiredInterval' by making sure that expected_output overlaps I.
            // Otherwise no work for that interval will be necessary.
            EXCEPTION_ASSERTX (expected_output & Signal::Interval(wanted_output.first, wanted_output.first+1),
                               boost::format("actual_output = %1%, x = %2%")
                               % expected_output % wanted_output);

            // Compare required_input to what's available in the sources
            missing_input &= required_input;

            // If there are no sources
            if (0==out_degree(u, g))
              {
                // Then this operation must specify sample rate and number of
                // samples for this to be a valid read. Otherwise the signal is
                // undefined.
                Signal ::OperationDesc::Extent x = o->extent ();
                if (!x.number_of_channels.is_initialized () || !x.sample_rate.is_initialized ())
                  {
                    // "Undefined signal. No sources and no extent"
                    missing_input = Signal::Interval::Interval_ALL;
                  }
              }

            // If nothing is missing and this engine supports this operation
            if (missing_input.empty ())
              {
                Signal::Operation::Ptr operation = o->createOperation (params.engine.get ());

                if (operation)
                  {
                    // Create a task
                    std::vector<Step::Ptr> children;

                    BOOST_FOREACH(GraphEdge e, out_edges(u, g))
                      {
                        GraphVertex v = target(e,g);
                        children.push_back (g[v]);
                      }

                    *task = Task::Ptr(new Task(step, children, operation, expected_output, required_input));
                  }
              }

            // Even if this engine doesn't support this operation it should
            // still update 'needed' so that it can compute what's
            // needed in the children, who might support this engine.
            return required_input;
          }
        catch (const boost::exception& x)
          {
            x   << Step::crashed_step(g[u]);

            try
              {
                Signal::Processing::IInvalidator::Ptr i = step->mark_as_crashed_and_get_invalidator();
                step.unlock ();
                if (i) read1(i)->deprecateCache (Signal::Intervals::Intervals_ALL);
              }
            catch(const std::exception& y)
              {
                x << unexpected_exception_info(boost::current_exception());
              }

            throw;
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
                Signal::IntervalType preferred_size,
                Workers::Ptr /*workers*/,
                Signal::ComputingEngine::Ptr engine) const
{
    DEBUGINFO TaskTimer tt(boost::format("FirstMissAlgorithm %s %p") % (engine?vartype(*engine):"Signal::ComputingEngine*") % engine.get ());
    Graph g; ReverseGraph::reverse_graph (straight_g, g);
    GraphVertex target = ReverseGraph::find_first_vertex (g, straight_g[straight_target]);

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
        Signal::ComputingEngine::Ptr c(new Signal::ComputingCpu);
        Task::Ptr t1 = schedule.getTask(g, v, Signal::Interval(20,30), 25, Interval::IntervalType_MAX, Workers::Ptr(), c);
        Task::Ptr t2 = schedule.getTask(g, v, Signal::Interval(10,24) | Signal::Interval(26,30), 25, Interval::IntervalType_MAX, Workers::Ptr(), c);


        // Verify output
        EXCEPTION_ASSERT(t1);
        EXCEPTION_ASSERT(t2);
        EXCEPTION_ASSERT_EQUALS(read1(t1)->expected_output(), Interval(20,30));
        EXCEPTION_ASSERT_EQUALS(read1(t2)->expected_output(), Interval(10, 20));

        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), Signal::Intervals::Intervals_ALL);
        EXCEPTION_ASSERT_EQUALS(~Signal::Intervals(10,30), read1(step)->not_started());

        // Verify that the output objects can be used
        write1(t1)->run();
        write1(t2)->run();
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), read1(step)->not_started());
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), ~Signal::Intervals(10,30));
    }

    // It should let missing_in_target override out_of_date in the given vertex
}


} // namespace Processing
} // namespace Signal
