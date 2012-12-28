#include "unittest.h"

#include "test/implicitordering.h"
#include "test/stdlibtest.h"
#include "test/tasktimertiming.h"
#include "tfr/freqaxis.h"
#include "tools/support/brushpaintkernel.h"

#include <stdio.h>
#include <exception>

#include <TaskTimer.h>

using namespace std;

namespace Test {

int UnitTest::
        test()
{
    try {
        TaskTimer tt("Tests ...");

        Test::ImplicitOrdering::test ();
        Test::Stdlibtest::test ();
        Test::TaskTimerTiming::test ();
        Tfr::FreqAxis::test ();
        Gauss::test ();

        printf("OK\n");
        return 0;

    } catch (const exception& x) {
        printf("%s\n", x.what());
        return 1;
    }
}

} // namespace Test
