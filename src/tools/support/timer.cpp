#include "timer.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

using namespace boost::posix_time;

namespace Tools {
namespace Support {

Timer::Timer()
{
    restart();
}


void Timer::restart()
{
#ifdef _MSC_VER
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    start_ = li.QuadPart;
#else
    start_ = microsec_clock::local_time();
#endif
}


double Timer::elapsed() const
{
#ifdef _MSC_VER
    LARGE_INTEGER li;
    static double PCfreq = 1;
    for(static bool doOnce=true;doOnce;doOnce=false)
    {
        QueryPerformanceFrequency(&li);
        PCfreq = double(li.QuadPart);
    }
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-start_)/PCfreq;
#else
    time_duration diff = microsec_clock::local_time() - start_;
    return diff.total_microseconds() * 1e-6;
#endif
}


} // namespace Support
} // namespace Tools
