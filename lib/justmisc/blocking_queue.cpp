#include "blocking_queue.h"
#include "exceptionassert.h"
#include "tasktimer.h"

#include <future>

using namespace std;

namespace JustMisc {

void blocking_queue_test::
        test ()
{
    // It should provide a thread safe solution to the multiple consumer-multiple producer pattern.
    {
        typedef blocking_queue<function<int()>> queue;
        queue q;

        auto producer = [&q](int a, int b) {
            for (int i=a; i<b; i++)
                q.push ([i]() { return i; });
        };

        auto consumer = [&q]() {
            int S = 0;

            try {
                while (true)
                    S += q.pop ()();
            } catch (queue::abort_exception) {}

            return S;
        };

        vector<thread> producers;
        vector<future<int>> consumers;
        for (int i=0; i<10; i++)
        {
            producers.push_back (thread(producer, 100*i, 100*i + 100));
            consumers.push_back (async(launch::async, consumer));
        }

        for (thread& t : producers)
            t.join ();

//        q.abort_on_empty ();

        int S = 0;
        for (future<int>& f : consumers)
        {
            int v = f.get ();
            EXCEPTION_ASSERT_LESS(1000, v);
            S += v;
        }

        EXCEPTION_ASSERT_EQUALS(S, (999-0)/2.0 * 1000);
    }

    // pop_for
    {
        typedef blocking_queue<int> queue;
        queue q;

        auto a = async(launch::async, [&q]() {
            EXCEPTION_ASSERT_EQUALS(q.pop_for (chrono::duration<double>(0.0001)), 0);
            EXCEPTION_ASSERT_EQUALS(q.pop_for (chrono::duration<double>(0.1)), 2);
            EXCEPTION_ASSERT_EQUALS(q.pop_for (chrono::duration<double>(0.0001)), 0);
        });

        this_thread::sleep_for (chrono::duration<double>(0.001));

        q.push (2);
        a.get();
    }

    // clear
    {
        typedef blocking_queue<int> queue;
        queue q;

        q.push (2);
        q.push (4);

        EXCEPTION_ASSERT(!q.empty ());
        auto q2 = q.clear ();
        EXCEPTION_ASSERT(q.empty ());
        EXCEPTION_ASSERT_EQUALS(q2.size (), 2u);
    }
}

} // namespace JustMisc
