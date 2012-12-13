#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"
<<<<<<< HEAD
#include "tools/support/timer.h"
#include "signal/buffer.h"
=======
#include "signal/dag/node.h"
#include "signal/dag/dagcommand.h"
#include "tools/support/timer.h"
>>>>>>> Stubbed FilterDescs for BlockFilter and AudiofileDesc for Audiofile.

#include <stdio.h>
#include <exception>

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
        Tools::Support::Timer timer;

        RUNTEST(Test::ImplicitOrdering);
        RUNTEST(Test::Stdlibtest);
        RUNTEST(Test::TaskTimerTiming);
        RUNTEST(Signal::Buffer);
        RUNTEST(Tfr::FreqAxis);
        RUNTEST(Gauss);
        RUNTEST(Signal::Dag::Node);
        RUNTEST(Signal::Dag::ICommand);

    } catch (const exception& x) {
        printf("\n%s\n\n", x.what());
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace Test
