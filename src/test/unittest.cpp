#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"
#include "tools/support/timer.h"
#include "signal/buffer.h"

#include <stdio.h>
#include <exception>

#include <TaskTimer.h>

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
    } catch (const exception& x) {
        printf("%s\n", x.what());
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace Test
