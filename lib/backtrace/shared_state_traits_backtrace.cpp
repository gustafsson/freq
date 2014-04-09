#include "shared_state_traits_backtrace.h"
#include "barrier.h"
#include "exceptionassert.h"
#include "trace_perf.h"

#include <future>

using namespace std;

class A {
public:
    struct shared_state_traits: shared_state_traits_backtrace {
        double timeout() { return 0.002; }
    };
};


void shared_state_traits_backtrace::
        test()
{
#ifndef SHARED_STATE_NO_TIMEOUT
    // shared_state can be extended with type traits to get, for instance,
    //  - backtraces on deadlocks from all participating threads,
    {
        typedef shared_state<A> ptr;
        ptr a{new A};
        ptr b{new A};

        spinning_barrier barrier(2);

        std::function<void(ptr,ptr)> m = [&barrier](ptr p1, ptr p2) {
            try {
                auto w1 = p1.write ();
                barrier.wait ();
                auto w2 = p2.write ();

                // never reached
                EXCEPTION_ASSERT(false);
            } catch (lock_failed& x) {
                // cheeck that a backtrace was embedded into the lock_failed exception
                const Backtrace* backtrace = boost::get_error_info<Backtrace::info>(x);
                EXCEPTION_ASSERT(backtrace);
            }
        };

        // Lock a and b in opposite order in f1 and f2
        future<void> f1 = async(launch::async, [&](){ m (b, a); });
        future<void> f2 = async(launch::async, [&](){ m (a, b); });

        f1.get ();
        f2.get ();
    }
#endif

    // shared_state can be extended with type traits to get, for instance,
    //  - warnings on locks that are held too long.
    {
        shared_state<A> a{new A};

        bool did_report = false;

        a.traits ()->exceeded_lock_time = [&did_report](float){ did_report = true; };

        auto w = a.write ();

        // Wait to make VerifyExecutionTime detect that the lock was kept too long
        this_thread::sleep_for (chrono::milliseconds{10});

        EXCEPTION_ASSERT(!did_report);
        w.unlock ();
        EXCEPTION_ASSERT(did_report);

        {
            int N = 10000;

            TRACE_PERF("warnings on locks that are held too long should cause a low overhead");
            for (int i=0; i<N; i++)
            {
                a.write ();
                a.read ();
            }
        }
    }
}
