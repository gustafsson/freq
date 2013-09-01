#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"
#include "adapters/microphonerecorder.h"
#include "signal/buffer.h"
#include "signal/cache.h"
#include "signal/dag/dagcommand.h"
#include "signal/dag/node.h"
#include "signal/dag/scheduler.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/chain.h"
#include "signal/processing/dag.h"
#include "signal/processing/firstmissalgorithm.h"
#include "signal/processing/graphinvalidator.h"
#include "signal/processing/sleepschedule.h"
#include "signal/processing/step.h"
#include "signal/processing/targetmarker.h"
#include "signal/processing/targetneeds.h"
#include "signal/processing/targets.h"
#include "signal/processing/targetschedule.h"
#include "signal/processing/task.h"
#include "signal/processing/worker.h"
#include "signal/processing/workers.h"
#include "signal/oldoperationwrapper.h"
#include "signal/operationwrapper.h"
#include "tools/commands/appendoperationdesccommand.h"
#include "tools/recordmodel.h"
#include "tools/support/renderoperation.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/blockfilter.h"
#include "heightmap/chunkblockfilter.h"
#include "adapters/playback.h"

// gpumisc units
#include "backtrace.h"
#include "exceptionassert.h"
#include "glinfo.h"
#include "prettifysegfault.h"
#include "volatileptr.h"

// gpumisc tool
#include "TaskTimer.h"
#include "timer.h"

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

        RUNTEST(Backtrace);
        RUNTEST(ExceptionAssert);
        RUNTEST(glinfo);
        RUNTEST(PrettifySegfault);
        RUNTEST(VolatilePtrTest);
        RUNTEST(Test::ImplicitOrdering);
        RUNTEST(Test::Stdlibtest);
        RUNTEST(Test::TaskTimerTiming);
        RUNTEST(Adapters::MicrophoneRecorderDesc);
        RUNTEST(Signal::Buffer);
        RUNTEST(Signal::BufferSource);
        RUNTEST(Tfr::FreqAxis);
        RUNTEST(Gauss);
        RUNTEST(Signal::Cache);
        RUNTEST(Signal::Dag::Node);
        RUNTEST(Signal::Dag::ICommand);
        RUNTEST(Signal::Dag::Scheduler);
        RUNTEST(Signal::Processing::Bedroom);
        RUNTEST(Signal::Processing::Chain);
        RUNTEST(Signal::Processing::Dag);
        RUNTEST(Signal::Processing::FirstMissAlgorithm);
        RUNTEST(Signal::Processing::GraphInvalidator);
        RUNTEST(Signal::Processing::SleepSchedule);
        RUNTEST(Signal::Processing::Step);
        RUNTEST(Signal::Processing::TargetMarker);
        RUNTEST(Signal::Processing::TargetNeeds);
        RUNTEST(Signal::Processing::Targets);
        RUNTEST(Signal::Processing::TargetSchedule);
        RUNTEST(Signal::Processing::Task);
        RUNTEST(Signal::Processing::Worker);
        RUNTEST(Signal::Processing::Workers);
        RUNTEST(Signal::OldOperationWrapper);
        RUNTEST(Signal::OldOperationDescWrapper);
        RUNTEST(Signal::OperationDescWrapper);
        RUNTEST(Tools::Commands::AppendOperationDescCommand);
        RUNTEST(Tools::RecordModel);
        RUNTEST(Tools::Support::RenderOperationDesc);
        RUNTEST(Heightmap::ChunkToBlock);
        RUNTEST(Heightmap::TfrMap);
        RUNTEST(Heightmap::CreateChunkBlockFilter);
        RUNTEST(Adapters::Playback);

    } catch (const exception& x) {
        TaskInfo(boost::format("%s") % boost::diagnostic_information(x));
        printf("\n FAILED in %s::test()\n\n", lastname.c_str ());
        return 1;
    } catch (...) {
        TaskInfo(boost::format("Not an std::exception\n%s") % boost::current_exception_diagnostic_information ());
        printf("\n FAILED in %s::test()\n\n", lastname.c_str ());
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace Test
