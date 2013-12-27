#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "test/randombuffer.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"
#include "adapters/microphonerecorder.h"
#include "signal/buffer.h"
#include "signal/cache.h"
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
#include "tfr/stftdesc.h"
#include "tfr/dummytransform.h"
#include "filters/envelope.h"
#include "tools/commands/appendoperationdesccommand.h"
#include "tools/openfilecontroller.h"
#include "tools/openwatchedfilecontroller.h"
#include "tools/recordmodel.h"
#include "tools/support/audiofileopener.h"
#include "tools/support/chaininfo.h"
#include "tools/support/renderoperation.h"
#include "tools/support/renderviewupdateadapter.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/blockfilter.h"
#include "heightmap/chunkblockfilter.h"
#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/tfrmappings/cwtblockfilter.h"
#include "heightmap/tfrmappings/waveformblockfilter.h"
#include "heightmap/render/renderset.h"
#include "adapters/playback.h"

// gpumisc units
#include "backtrace.h"
#include "exceptionassert.h"
#include "geometricalgebra.h"
#include "glinfo.h"
#include "glprojection.h"
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
        RUNTEST(GeometricAlgebra);
        RUNTEST(glinfo);
        RUNTEST(glProjection);
        RUNTEST(neat_math);
        RUNTEST(PrettifySegfault);
        RUNTEST(VolatilePtrTest);
        RUNTEST(Test::ImplicitOrdering);
        RUNTEST(Test::Stdlibtest);
        RUNTEST(Test::TaskTimerTiming);
        RUNTEST(Test::RandomBuffer);
        RUNTEST(Adapters::MicrophoneRecorderDesc);
        RUNTEST(Signal::Buffer);
        RUNTEST(Signal::BufferSource);
        RUNTEST(Tfr::FreqAxis);
        RUNTEST(Gauss);
        RUNTEST(Timer);
        RUNTEST(Signal::Cache);
        RUNTEST(Signal::Intervals);
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
        RUNTEST(Tfr::StftDesc);
        RUNTEST(Tfr::DummyTransform);
        RUNTEST(Tfr::DummyTransformDesc);
        RUNTEST(Filters::EnvelopeDesc);
        RUNTEST(Tools::Commands::AppendOperationDescCommand);
        RUNTEST(Tools::OpenfileController);
        RUNTEST(Tools::OpenWatchedFileController);
        RUNTEST(Tools::RecordModel);
        RUNTEST(Tools::Support::AudiofileOpener);
        RUNTEST(Tools::Support::ChainInfo);
        RUNTEST(Tools::Support::RenderOperationDesc);
        RUNTEST(Tools::Support::RenderViewUpdateAdapter);
        RUNTEST(Tools::Support::HeightmapProcessingPublisher);
        RUNTEST(Heightmap::Block);
        RUNTEST(Heightmap::BlockCache);
        RUNTEST(Heightmap::BlockLayout);
        RUNTEST(Heightmap::ChunkToBlock);
        RUNTEST(Heightmap::Render::RenderSet);
        RUNTEST(Heightmap::TfrMapping);
        RUNTEST(Heightmap::VisualizationParams);
        RUNTEST(Heightmap::ChunkBlockFilter);
        RUNTEST(Heightmap::ChunkBlockFilterDesc);
        RUNTEST(Heightmap::TfrMappings::StftBlockFilter);
        RUNTEST(Heightmap::TfrMappings::StftBlockFilterDesc);
        RUNTEST(Heightmap::TfrMappings::CwtBlockFilter);
        RUNTEST(Heightmap::TfrMappings::CwtBlockFilterDesc);
        RUNTEST(Heightmap::TfrMappings::WaveformBlockFilter);
        RUNTEST(Heightmap::TfrMappings::WaveformBlockFilterDesc);
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
