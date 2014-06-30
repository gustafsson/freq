#include "unittest.h"

#include "../lib/backtrace/unittest.h"
#include "../lib/justmisc/justmisc-unittest.h"
#include "../lib/gpumisc/unittest.h"
#include "../lib/signal/signal/unittest.h"
#include "../lib/tfr/tfr/unittest.h"
#include "../lib/heightmap/heightmap/unittest.h"

// sonicawe
#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "test/randombuffer.h"
#include "test/printbuffer.h"
#include "tools/support/brushpaintkernel.h"
#include "filters/selection.h"
#include "filters/envelope.h"
#include "filters/normalize.h"
#include "filters/rectangle.h"
#include "filters/timeselection.h"
#include "tools/support/audiofileopener.h"
#include "tools/support/csvfileopener.h"
#include "tools/support/chaininfo.h"
#include "tools/support/operation-composite.h"
#include "tools/support/renderoperation.h"
#include "tools/support/renderviewupdateadapter.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "tools/support/workercrashlogger.h"
#include "tools/support/computerms.h"
#include "tools/commands/appendoperationdesccommand.h"
#include "tools/openfilecontroller.h"
#include "tools/openwatchedfilecontroller.h"
#include "tools/recordmodel.h"
#include "tools/applicationerrorlogcontroller.h"
#include "adapters/playback.h"
#include "adapters/microphonerecorder.h"
#include "filters/absolutevalue.h"

// common backtrace tools
#include "timer.h"
#include "tasktimer.h"
#include "trace_perf.h"
#include "demangle.h"

#include <stdio.h>
#include <exception>

#include <boost/exception/diagnostic_information.hpp>

using namespace std;

namespace Test {

string lastname;

#define RUNTEST(x) do { \
        TaskTimer tt("%s", #x); \
        lastname = #x; \
        x::test (); \
    } while(false)

int UnitTest::
        test()
{
    try {
        Timer(); // Init performance counting
        TaskTimer tt("Running tests");

        trace_perf::add_database_path("../lib/backtrace/trace_perf");
        trace_perf::add_database_path("../lib/gpumisc/trace_perf");
        trace_perf::add_database_path("../lib/heightmap/trace_perf");
        trace_perf::add_database_path("../lib/tfr/trace_perf");

        RUNTEST(BacktraceTest::UnitTest);
        RUNTEST(JustMisc::UnitTest);
        RUNTEST(gpumisc::UnitTest);
        RUNTEST(Signal::UnitTest);
        RUNTEST(Tfr::UnitTest);
        RUNTEST(Heightmap::UnitTest);

        RUNTEST(Test::ImplicitOrdering);
        RUNTEST(Test::Stdlibtest);
        RUNTEST(Test::TaskTimerTiming);
        RUNTEST(Test::RandomBuffer);
        RUNTEST(Test::PrintBuffer);
        RUNTEST(Signal::Buffer);
        RUNTEST(Signal::BufferSource);
        RUNTEST(Tfr::FreqAxis);
        RUNTEST(Gauss);
        // PortAudio complains if testing Microphone in the end
        RUNTEST(Adapters::MicrophoneRecorderDesc);
        RUNTEST(Filters::Selection);
        RUNTEST(Filters::EnvelopeDesc);
        RUNTEST(Filters::Normalize);
        RUNTEST(Filters::Rectangle);
        RUNTEST(Filters::TimeSelection);
        RUNTEST(Tools::OpenfileController);
        RUNTEST(Tools::OpenWatchedFileController);
        RUNTEST(Tools::RecordModel);
        RUNTEST(Tools::Support::AudiofileOpener);
        RUNTEST(Tools::Support::CsvfileOpener);
        RUNTEST(Tools::Support::ChainInfo);
        RUNTEST(Tools::Support::OperationCrop);
        RUNTEST(Tools::Support::RenderOperationDesc);
        RUNTEST(Tools::Support::RenderViewUpdateAdapter);
        RUNTEST(Tools::Support::HeightmapProcessingPublisher);
        RUNTEST(Tools::Support::WorkerCrashLogger);
        RUNTEST(Tools::Support::ComputeRmsDesc);
        RUNTEST(Tools::Commands::AppendOperationDescCommand);
        RUNTEST(Tools::ApplicationErrorLogController);
        RUNTEST(Adapters::Playback);
        RUNTEST(Filters::AbsoluteValueDesc);

    } catch (const ExceptionAssert& x) {
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
        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("%s\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % vartype(x) % boost::diagnostic_information(x) % lastname ).c_str());
        fflush(stderr);
        return 1;
    } catch (...) {
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

} // namespace Test
