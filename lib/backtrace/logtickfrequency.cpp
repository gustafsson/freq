#include "logtickfrequency.h"
#include "log.h"
#include "exceptionassert.h"

LogTickFrequency::
        LogTickFrequency(std::string title, double loginterval)
    :
      title_(title),
      loginterval_(loginterval),
      T_(0.),
      timer_(false),
      ticks_(-1)
{
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0.,loginterval);
}


LogTickFrequency::
        ~LogTickFrequency()
{
//    T_ += timer_.elapsedAndRestart ();
//    double f = ticks_ / T_;
//    if (ticks_<0)
//        f = 0;
//    else
//        Log("%s: %g ticks/s") % title_ % f;
}


bool LogTickFrequency::
        tick(bool lognow)
{
    if (ticks_<0) {
        timer_.restart ();
        ticks_=0;
        return false;
    }

    ticks_++;
    T_ += timer_.elapsedAndRestart ();
    if (T_ > loginterval_)
    {
        if (lognow)
            log();
        return true;
    }
    return false;
}


double LogTickFrequency::
        hz(bool reset)
{
    double f = ticks_ / T_;
    if (reset)
    {
        T_ = 0.;
        ticks_ = 0;
    }
    return f;
}


void LogTickFrequency::
        log()
{
    Log("%s: %g/s") % title_ % hz();
}
