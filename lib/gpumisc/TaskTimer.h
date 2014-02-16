#pragma once

#include <boost/date_time/posix_time/ptime.hpp>
#include <stdarg.h>
#if defined(__cplusplus) && !defined(__CUDACC__)
    #include <ostream>
#endif
#include <boost/utility.hpp>
#include <boost/format.hpp>

/**
TaskTimer is a convenient timing tool used to measure how long time
it takes to execute a code block. This is illustrated by this
example:

<p><code>
{
    TaskTimer tt("Doing this slow thing");
    doSlowThing();
}

</code><p>
or

<p><code>
TIME_TASK( doSlowThing() );

</code><p>
in which the latter expands to the first using the macro TIME_TASK.
When this is run the following will be seen on cout:

<p><code>
Doing this slow thing... done in 100 ms.

</code><p>
Where "done in 100 ms." will be sent to cout when doSlowThing has
returned, and TaskTimer tt is going out of scope.

<p>
Anoother example:

<p><code>
{
    TaskTimer tt("Doing this slow thing");
    initialize();
    for (int i=0; i<9; i++) {
        tt.partlyDone();
        doSlowThing();
    }
}

</code><p>
Where tt.partlyDone() will output an extra dot "." for each call.
*/
class TaskTimer: private boost::noncopyable {
public:
    enum LogLevel {
        LogVerbose = 0,
        LogDetailed = 1,
        LogSimple = 2
    };

    TaskTimer(LogLevel logLevel, const char* task, ...);
    TaskTimer(bool, LogLevel logLevel, const char* task, va_list args);
    TaskTimer(const char* task, ...);
    TaskTimer(bool, const char* task, va_list args);
    TaskTimer(const boost::format& fmt);
    TaskTimer();
    ~TaskTimer();

    static void this_thread_quit();

    void info(const char* taskInfo, ...);
    void partlyDone();
    void suppressTiming();
    float elapsedTime();

    #if defined(__cplusplus) && !defined(__CUDACC__)
        static void setLogLevelStream( LogLevel logLevel, std::ostream* str );
        static bool isEnabled(LogLevel logLevel);
    #endif

    static bool enabled();
    static void setEnabled( bool );

private:
    boost::posix_time::ptime startTime;

#ifdef _MSC_VER
        __int64 hpcStart;
#endif

    unsigned numPartlyDone;
    bool is_unwinding;
    bool suppressTimingInfo;
    LogLevel logLevel;

    TaskTimer* upperLevel; // obsolete

    //TaskTimer& getCurrentTimer();
    void init(LogLevel logLevel, const char* task, va_list args);
    void initEllipsis(LogLevel logLevel, const char* f, ...);
    void vinfo(const char* taskInfo, va_list args);
    void logprint(const char* txt);
    bool printIndentation();
};

class TaskInfo: private boost::noncopyable {
public:
    TaskInfo(const char* task, ...);
    TaskInfo(const boost::format&);
    ~TaskInfo();

    TaskTimer& tt() { return *tt_; }
private:
    TaskTimer* tt_;
};

#define TaskLogIfFalse(X) if (false == (X)) TaskInfo("! Not true: %s", #X)
#define TIME(x) do { TaskTimer tt("%s", #x); x; } while(false)
