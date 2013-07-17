#include "tasktimertiming.h"
#include "tools/support/timer.h"

#include "TaskTimer.h"
#include "ThreadChecker.h"
#include "exceptionassert.h"

#include <sstream>

namespace Test {

void TaskTimerTiming::
        test()
{
    std::stringstream dummy;
    try
    {
        Tools::Support::Timer t;
        TaskTimer::setLogLevelStream (TaskTimer::LogSimple, &dummy);
        TaskTimer::setLogLevelStream (TaskTimer::LogVerbose, &dummy);
        TaskTimer::setLogLevelStream (TaskTimer::LogDetailed, &dummy);

        {
            TaskTimer tt("Timing tasktimer");
        }
        {
            TaskTimer tt("Timing loop");
            for (unsigned N = 1000; N; --N)
            {
            }
        }
        {
            TaskTimer tt("Timing threadchecker");
            for (unsigned N = 1000; N; --N)
            {
                ThreadChecker tc;
            }
        }
        // Ubuntu, debug build of both gpumisc and sonicawe
        //00:12:20.787 Timing tasktimer... done in 4.0 us.
        //00:12:20.788 Timing loop... done in 6.0 us.
        //00:12:20.788 Timing threadchecker... done in 37.0 us.
        TaskTimer::setLogLevelStream (TaskTimer::LogSimple, &std::cout);
        TaskTimer::setLogLevelStream (TaskTimer::LogVerbose, &std::cout);
        TaskTimer::setLogLevelStream (TaskTimer::LogDetailed, &std::cout);
        float T = t.elapsed ();
        EXCEPTION_ASSERTX( T < 0.0006, boost::format("T was %1%") % T);
    }
    catch( ... )
    {
        TaskTimer::setLogLevelStream (TaskTimer::LogSimple, &std::cout);
        TaskTimer::setLogLevelStream (TaskTimer::LogVerbose, &std::cout);
        TaskTimer::setLogLevelStream (TaskTimer::LogDetailed, &std::cout);
        throw;
    }
}

} // namespace Test
