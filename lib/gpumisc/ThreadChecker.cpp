#include "ThreadChecker.h"
#include "stringprintf.h"
#include "exceptionassert.h"

#include <stdexcept>

#include "TaskTimer.h" // TODO remove

ThreadChecker::ThreadChecker()
:   startThread( getCurrentThread() )
{}

ThreadChecker::ThreadChecker( void * id )
:   startThread( id )
{}

bool
ThreadChecker::isSameThread() const
{
    return 0==startThread || startThread == getCurrentThread();
}

void
ThreadChecker::throwIfNotSame( const char *funcsig) const
{
    EXCEPTION_ASSERT( isSameThread() );

    if( ! isSameThread() )
        throw DifferentThreadException( printfstring( "%s can only be called "
            "from the same thread as the instance was created in. First "
            "created in thread %p, now called in thread %p.",
            funcsig, startThread, getCurrentThread()));
}


#ifdef THREADCHECKER_NO_CHECK

    void*
    ThreadChecker::getCurrentThread()
    {
        return 0;
    }

#else // THREADCHECKER_NO_CHECK

    #if defined(_WIN32)

        #include <windows.h>

        void*
        ThreadChecker::getCurrentThread()
        {
            return (void*)(INT_PTR)(DWORD)GetCurrentThreadId();
        }

    #else

        #include <pthread.h>

        void*
        ThreadChecker::getCurrentThread()
        {
            return (void*)pthread_self();
        }

    #endif

#endif // ifndef THREADCHECKER_NO_CHECK
