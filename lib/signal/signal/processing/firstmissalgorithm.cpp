#include "firstmissalgorithm.h"

#include "reversegraph.h"
#include "signal/buffersource.h"

#include "tasktimer.h"
#include "expectexception.h"
#include "neat_math.h"
#include "log.h"

#include <boost/foreach.hpp>
#include <boost/graph/breadth_first_search.hpp>

//#define DEBUGINFO
#define DEBUGINFO if(0)

using namespace boost;

namespace Signal {
namespace Processing {

std::mutex debuginfo_firstmissingalgorithm;

typedef std::map<GraphVertex, Signal::Intervals> NeededSamples;


struct ScheduleParams {
    Signal::ComputingEngine::ptr engine;
    Signal::IntervalType preferred_size;
    Signal::IntervalType center;
};


class find_missing_samples: public default_bfs_visitor {
public:
    find_missing_samples(NeededSamples needed, Task* output_task, ScheduleParams schedule_params)
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
            missing_input |= ~Step::cache (g[v])->samplesDesc();
          }

        Intervals required_input = try_create_task(u, g, missing_input);

        // Update sources with needed samples
        BOOST_FOREACH(GraphEdge e, out_edges(u, g))
          {
            GraphVertex v = target(e,g);
            DEBUGINFO
            {
                Intervals not_started = g[v].read ()->not_started ();
                if (not_started & required_input)
                    TaskInfo(format("firstmissing: %s needs %s from %s to create a task. %s is not started.")
                                     % Step::operation_desc (g[u])->toString ().toStdString ()
                                     % required_input
                                     % Step::operation_desc (g[v])->toString ().toStdString ()
                                     % not_started);
            }
            needed[v] |= required_input;
          }
      }


    Signal::Intervals try_create_task(GraphVertex u, const Graph & g, const Signal::Intervals& all_missing_input)
    {
        // TODO could be more greedy, example: change parameters during recording

        // No other thread is allowed to reach the same conclusion about what needs to be done.
        // So the lock has to be kept from checking what's needed all the way until the task has
        // been registered in step->running_tasks.
        auto step_ptr = g[u];
        // Lock while studying what's needed. Lock outside try-catch to use the lock in the catch-block as well.
        auto step = step_ptr.write ();

        try
        {
            Signal::OperationDesc::const_ptr o = Step::operation_desc (step_ptr);
            Signal::Intervals I = needed[u] & step->not_started ();
            DEBUGINFO TaskInfo ti(format("firstmissing: %s, I: %s, all_missing_input: %s")
                                  % Step::operation_desc (step_ptr)->toString ().toStdString ()
                                  % I % all_missing_input);
            if ((0)) Log ("needed[u]: %s, step->not_started (): %s") % needed[u] % step->not_started ();

            // figure out what's needed from the sources
            Signal::Intervals all_required_input;
            Signal::Interval expected_output;
            for (auto i : I) all_required_input |= o->requiredInterval (i, &expected_output);

            // if the sources have some things available but not the whole wanted_output, take that instead
            Signal::Intervals not_wanted_output;
            for (auto i : all_missing_input) not_wanted_output |= o->affectedInterval (i);
            I -= not_wanted_output;

            // For all intervals that are needed from this step, compute what we need from sources
            while (I)
            {
                // params.preferred_size is just a preferred update size, not a required update size.
                // Accept whatever requiredInterval sets as expected_output
                Signal::Interval wanted_output = I.fetchInterval(params.preferred_size, params.center);

                // if it's less than 2*params.preferred_size left to finish, take that instead
                Signal::Interval wanted_output2 = I.fetchInterval(clamped_add(params.preferred_size,params.preferred_size), params.center);
                if (wanted_output.count () > wanted_output2.count ()/2)
                    wanted_output = wanted_output2;

                I -= wanted_output; // reduce I in search for something to work on

                // figure out expected_output to ensure a valid output range
                Signal::Interval required_input = o->requiredInterval (wanted_output, &expected_output);
                all_required_input |= required_input; // update all_required_input if o has changed state since first requiredInterval

                if (!expected_output)
                {
                    // Can't compute the requested output. Not because something in
                    // particular is missing from the sourcs, the operation isn't
                    // just ready yet. The worker threads should wait for a wakeup.
                    DEBUGINFO Log("wanted_output = %s, expected_output = %s, required_input = %s")
                            % wanted_output % expected_output % required_input;
                    continue;
                }

                //check that OperationDesc returned that it needed at least something
                EXCEPTION_ASSERTX(required_input, o->toString ().toStdString ());
                // check for valid 'requiredInterval' by making sure that expected_output overlaps I.
                // Otherwise no work for that interval will be necessary.
                EXCEPTION_ASSERTX (expected_output & Signal::Interval(wanted_output.first, wanted_output.first+1),
                                   boost::format("actual_output = %1%, x = %2%")
                                   % expected_output % wanted_output);

                // Compare required_input to what's available in the sources
                Signal::Intervals missing_input = all_missing_input & required_input;

                DEBUGINFO Log("wanted_output = %s, expected_output = %s, required_input = %s, missing_input = %s")
                        % wanted_output % expected_output % required_input % missing_input;

                // If nothing is missing
                if( !missing_input )
                {
                    // If there are no sources
                    if( 0==out_degree(u, g) )
                    {
                        // Then this operation must specify sample rate and number of
                        // samples for this to be a valid read. Otherwise the signal is
                        // undefined.
                        auto x = o->extent ();
                        if (!x.number_of_channels.is_initialized () || !x.sample_rate.is_initialized ())
                        {
                            // "Undefined signal. No sources and no extent"
                            return Signal::Interval::Interval_ALL;
                        }
                    }

                    Signal::Operation::ptr operation = o->createOperation (params.engine.get ());

                    // If this engine supports this operation
                    if (operation)
                    {
                        // Create a task
                        std::vector<Step::const_ptr> children;

                        BOOST_FOREACH(GraphEdge e, out_edges(u, g))
                        {
                            GraphVertex v = target(e,g);
                            children.push_back (g[v]);
                        }

                        *task = Task(step, g[u], children, operation, expected_output, required_input);
                        return Signal::Intervals(); // no need to compute further, we've got a task
                    }
                }
            } // while
            // Even if this engine doesn't support this operation it should
            // still update 'needed' so that it can compute what's
            // needed in the children, who might support this engine.
            return all_required_input;
        } // try
        catch (const boost::exception& x)
        {
            // Keep lock until the step has been marked as crashed
            IInvalidator::ptr i = step->mark_as_crashed_and_get_invalidator();
            step.unlock();

            // Propagate the exception but decorate it with some info
            x << Step::crashed_step(g[u]);

            try
            {
                if (i) i->deprecateCache (Signal::Intervals::Intervals_ALL);
            }
            catch(const std::exception& y)
            {
                x << unexpected_exception_info(boost::current_exception());
            }

            throw;
        }
    }

    NeededSamples needed;
    const ScheduleParams params;
    Task* task;
};


Task FirstMissAlgorithm::
        getTask(const Graph& straight_g,
                GraphVertex straight_target,
                Signal::Intervals needed,
                Signal::IntervalType center,
                Signal::IntervalType preferred_size,
                Signal::ComputingEngine::ptr engine) const
{
    DEBUGINFO std::unique_lock<std::mutex> l(debuginfo_firstmissingalgorithm);

    DEBUGINFO TaskTimer tt(boost::format("FirstMissAlgorithm %s %p") % (engine?vartype(*engine):"Signal::ComputingEngine*") % engine.get ());
    DEBUGINFO Log("needed = %s in %s") % needed % Step::operation_desc (straight_g[straight_target])->toString().toStdString();
    DEBUGINFO Log("params.preferred_size = %d, params.center = %d")
             % preferred_size % center;
    Graph g; ReverseGraph::reverse_graph (straight_g, g);
    GraphVertex target = ReverseGraph::find_first_vertex (g, straight_g[straight_target]);

    ScheduleParams schedule_params = { engine, preferred_size, center };

    NeededSamples needed_samples;
    needed_samples[target] = needed;


    Task task;
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
        Signal::OperationDesc::ptr od(new BufferSource(b));
        Step::ptr step(new Step(od));

        // Create a graph with only one vertex
        Graph g;
        GraphVertex v = g.add_vertex (step);


        // Schedule a task
        FirstMissAlgorithm schedule;
        Signal::ComputingEngine::ptr c(new Signal::ComputingCpu);
        Task t1 = schedule.getTask(g, v, Signal::Interval(20,30), 25, Interval::IntervalType_MAX, c);
        Task t2 = schedule.getTask(g, v, Signal::Interval(10,24) | Signal::Interval(26,30), 25, Interval::IntervalType_MAX, c);


        // Verify output
        EXCEPTION_ASSERT(t1);
        EXCEPTION_ASSERT(t2);
        EXCEPTION_ASSERT_EQUALS(t1.expected_output(), Interval(20,30));
        EXCEPTION_ASSERT_EQUALS(t2.expected_output(), Interval(10, 20));

        EXCEPTION_ASSERT_EQUALS(Step::cache (step)->samplesDesc(), Signal::Intervals());
        EXCEPTION_ASSERT_EQUALS(Signal::Intervals(10,30), ~step.read ()->not_started());

        // Verify that the output objects can be used
        t1.run();
        t2.run();
        EXCEPTION_ASSERT_EQUALS(Step::cache (step)->samplesDesc(), ~step.read ()->not_started());
        EXCEPTION_ASSERT_EQUALS(Step::cache (step)->samplesDesc(), Signal::Intervals(10,30));
    }

    // It should let missing_in_target override out_of_date in the given vertex
}


} // namespace Processing
} // namespace Signal
