#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"
#include "signal/buffer.h"
#include "signal/cache.h"
#include "signal/dag/dagcommand.h"
#include "signal/dag/node.h"
#include "signal/dag/scheduler.h"
#include "tools/support/timer.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/blockfilter.h"
#include "heightmap/chunkblockfilter.h"
#include "volatileptr.h"
#include "adapters/playback.h"

#include <stdio.h>
#include <exception>

#include <boost/exception/diagnostic_information.hpp>

using namespace std;

namespace Test {

#define RUNTEST(x) do { \
        TaskTimer tt("%s", #x); \
        x::test (); \
    } while(false)

int UnitTest::
        test()
{
    try {
        Tools::Support::Timer(); // Init performance counting
        TaskTimer tt("Running tests");

        RUNTEST(Test::ImplicitOrdering);
        RUNTEST(Test::Stdlibtest);
        RUNTEST(Test::TaskTimerTiming);
        RUNTEST(Signal::Buffer);
        RUNTEST(Signal::BufferSource);
        RUNTEST(Tfr::FreqAxis);
        RUNTEST(Gauss);
        RUNTEST(Signal::Cache);
        RUNTEST(Signal::Dag::Node);
        RUNTEST(Signal::Dag::ICommand);
        RUNTEST(Signal::Dag::Scheduler);
        RUNTEST(Heightmap::ChunkToBlock);
        RUNTEST(Heightmap::TfrMap);
        RUNTEST(Heightmap::CreateChunkBlockFilter);
        RUNTEST(VolatilePtrTest);
        RUNTEST(Adapters::Playback);

    } catch (exception& x) {
        printf("\n%s\n\n", boost::diagnostic_information(x).c_str());
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace Test
