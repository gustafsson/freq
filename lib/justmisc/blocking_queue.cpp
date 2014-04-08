#include "blocking_queue.h"
#include "exceptionassert.h"
#include "tasktimer.h"

#include <future>

using namespace std;

void blocking_queue_test::
        test ()
{
    // It should provide a thread safe solution to the multiple consumer-multiple producer pattern.
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

    std::vector<std::thread> producers;
    std::vector<std::future<int>> consumers;
    for (int i=0; i<10; i++)
    {
        producers.push_back (thread(producer, 100*i, 100*i + 100));
        consumers.push_back (async(launch::async, consumer));
    }

    for (std::thread& t : producers)
        t.join ();

    q.abort_on_empty ();

    int S = 0;
    for (std::future<int>& f : consumers)
    {
        int v = f.get ();
        EXCEPTION_ASSERT_LESS(1000, v);
        S += v;
    }

    EXCEPTION_ASSERT_EQUALS(S, (999-0)/2.0 * 1000);
}
