#include "thread_pool.h"
#include "exceptionassert.h"
#include "expectexception.h"
#include "tasktimer.h"

#include <numeric>
#include <functional>

using namespace std;

namespace JustMisc {

thread_pool::
        thread_pool()
    :
      thread_pool(thread::hardware_concurrency ())
{}


thread_pool::
        thread_pool(int n)
    :
      threads_(n)
{
    for (thread& t: threads_)
    {
        t = thread(
                [this]()
                {
                    try {
                        while (true) {
                            auto task = queue_.pop ();
                            task();
                        }
                    } catch (decltype(queue_)::abort_exception) {}
                }
        );
    }
}


thread_pool::
        ~thread_pool()
{
    queue_.abort_on_empty ();
    queue_.clear (); // Any associated futures to a packaged_task will be notified if the task is destroyed prior to evaluation

    for (thread& t: threads_)
        t.join ();
}


void thread_pool::
        test()
{
    // It should consume tasks in multiple threads.
    {
        vector<future<int>> R;
        thread_pool pool;

        for (int i=0; i<1000; i++)
        {
            packaged_task<int()> task(
                    [i]()
                    {
                        return i;
                    }
            );

            R.push_back (task.get_future ());

            pool.addTask (move(task));

            pool.addTask (packaged_task<void()>(
                    []()
                    {
                        this_thread::sleep_for (chrono::duration<double>(0.0001));
                    }
            ));
        }

        int S = accumulate(R.begin (), R.end (), 0,
                [](int a, future<int>& b)
                {
                    return a + b.get();
                }
        );

        EXCEPTION_ASSERT_EQUALS(S, (999-0)/2.0 * 1000);
    }

    // Any associated futures to a packaged_task will be notified if the task is destroyed prior to evaluation
    {
        auto* task = new packaged_task<void()>(
                []()
                {
                    return;
                }
        );

        auto f = task->get_future();

        delete task;

        f.wait();

        EXPECT_EXCEPTION(std::future_error, f.get());
    }
}

} // namespace JustMisc
