#ifndef TOOLS_SUPPORT_TIMER_H
#define TOOLS_SUPPORT_TIMER_H

#ifndef _MSC_VER
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

namespace Tools {
namespace Support {

class Timer
{
public:
    Timer();

    void restart();
    double elapsed() const;

private:
#ifdef _MSC_VER
    __int64 start_;
#else
    boost::posix_time::ptime start_;
#endif
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_TIMER_H
