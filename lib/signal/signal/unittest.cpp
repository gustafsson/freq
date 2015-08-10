#include "unittest.h"

#include "signal/buffer.h"
#include "signal/buffersource.h"
#include "signal/cache.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/chain.h"
#include "signal/processing/dag.h"
#include "signal/processing/firstmissalgorithm.h"
#include "signal/processing/graphinvalidator.h"
#include "signal/processing/step.h"
#include "signal/processing/targetmarker.h"
#include "signal/processing/targetneeds.h"
#include "signal/processing/targets.h"
#include "signal/processing/targetschedule.h"
#include "signal/processing/task.h"
#include "signal/pollworker/pollworker.h"
#include "signal/pollworker/pollworkers.h"
#include "signal/operationwrapper.h"

// common backtrace tools
#include "timer.h"
#include "tasktimer.h"
#include "trace_perf.h"
#include "demangle.h"

#include <stdio.h>
#include <exception>

#include <boost/exception/diagnostic_information.hpp>

using namespace std;

namespace Signal {

string lastname;

#define RUNTEST(x) do { \
        TaskTimer tt("%s", #x); \
        lastname = #x; \
        x::test (); \
    } while(false)

int UnitTest::
        test(bool rethrow_exceptions)
{
    try {
        Timer(); // Init performance counting
        TaskTimer tt("Running tests");

        RUNTEST(Signal::Intervals);
        RUNTEST(Signal::Buffer);
        RUNTEST(Signal::BufferSource);
        RUNTEST(Signal::Cache);
        RUNTEST(Signal::Processing::Bedroom);
        RUNTEST(Signal::Processing::Dag);
        RUNTEST(Signal::Processing::FirstMissAlgorithm);
        RUNTEST(Signal::Processing::GraphInvalidator);
        RUNTEST(Signal::Processing::Step);
        RUNTEST(Signal::Processing::TargetMarker);
        RUNTEST(Signal::Processing::TargetNeeds);
        RUNTEST(Signal::Processing::Targets);
        RUNTEST(Signal::Processing::TargetSchedule);
        RUNTEST(Signal::Processing::Task);
        RUNTEST(Signal::PollWorker::PollWorker);
        RUNTEST(Signal::PollWorker::PollWorkers);
        RUNTEST(Signal::Processing::Chain); // Chain last
        RUNTEST(Signal::OperationDescWrapper);

    } catch (const ExceptionAssert& x) {
        if (rethrow_exceptions)
            throw;

        char const * const * f = boost::get_error_info<boost::throw_file>(x);
        int const * l = boost::get_error_info<boost::throw_line>(x);
        char const * const * c = boost::get_error_info<ExceptionAssert::ExceptionAssert_condition>(x);
        std::string const * m = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x);

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("%s:%d: %s. %s\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % (f?*f:0) % (l?*l:-1) % (c?*c:0) % (m?*m:0) % boost::diagnostic_information(x) % lastname ).c_str());
        fflush(stderr);
        return 1;
    } catch (const exception& x) {
        if (rethrow_exceptions)
            throw;

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("%s\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % vartype(x) % boost::diagnostic_information(x) % lastname ).c_str());
        fflush(stderr);
        return 1;
    } catch (...) {
        if (rethrow_exceptions)
            throw;

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("Not an std::exception\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % boost::current_exception_diagnostic_information () % lastname ).c_str());
        fflush(stderr);
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace BacktraceTest
