#pragma once

#include <stdexcept>

/**
 ThreadChecker is used in contexts that are only valid for one 
 thread. The constructing thread id is fetched in the constructor
 and checked against the current thread id in the member functions.
 
 @author johan.b.gustafsson@gmail.com
 */
class ThreadChecker {
public:
    /**
     DifferentThreadException is thrown by 
     ThreadChecker#throwIfNotSame if ID of the calling thread differs
     from the ID of the constructing thread.
    */
    class DifferentThreadException: public std::runtime_error {
    public:
        DifferentThreadException( const std::string& message )
        :   runtime_error( message ) {}
    };

    /**
     Creates a new ThreadChecker and remembers the ID of the calling
     thread.
    */
    ThreadChecker();

    /**
     Creates a new ThreadChecker which remembers the given thread ID.
     Typically used as ThreadChecker(0) which sais that any thread
     is compatible to this ThreadChecker.
      */
    ThreadChecker(void* id);

    /**
     ThreadChecker throws a DifferentThreadException upon destruction
     if the calling thread does not have the same ID as the thread 
     that created this instance of ThreadChecker.

     <p>This is not always true. If the '='-operator has been used,
     an exception will only be thrown if the calling thread does not
     have the same ID as the ThreadChecker that was assigned from.

     <p>To make this handy during debugging, put a breakpoint at the
     'throw DifferentThreadException' statement in the method
     #throwIfNotSame in ThreadChecker.cpp.
    */
    ~ThreadChecker() {
        throwIfNotSame( __FUNCTION__ );
    }

    void reset() { ThreadChecker b; startThread = b.startThread; }

    /**
     Returns true if the calling thread has the same ID as the thread
     that created this instance of ThreadChecker.
    */
    bool isSameThread() const;

    /**
     Throws a DifferentThreadException if the calling thread does not 
     have the same ID as the thread that created this instance of 
     ThreadChecker.
    */
    void throwIfNotSame( const char* funcsig ) const;

protected:
    /**
     startThread has the result of getCurrentThread after 
     construction.
    */
    void* startThread;

    /**
     Retrieves an ID of the calling thread. This is system dependent 
     but typecasted to void* before returning.
    */
    static void* getCurrentThread();
};
