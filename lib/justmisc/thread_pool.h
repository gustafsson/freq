#ifndef JUSTMISC_THREAD_POOL_H
#define JUSTMISC_THREAD_POOL_H

#include "blocking_queue.h"

#include <future>
#include <thread>

namespace JustMisc {

/**
 * @brief The thread_pool class should consume tasks in multiple threads.
 */
class thread_pool
{
public:
    thread_pool();
    thread_pool(int N);
    ~thread_pool();

    void addTask(std::packaged_task<void()>&& task)
    {
        queue_.push (std::move(task));
    }

    template<class F>
    void addTask(std::packaged_task<F()>&& task)
    {
        queue_.push (std::packaged_task<void()>(
                [task(std::move(task))] () mutable
                {
                    task();
                }
        ));
    }

private:
    blocking_queue<std::packaged_task<void()>> queue_;
    std::vector<std::thread> threads_;

public:
    static void test();
};

} // namespace JustMisc

#endif // JUSTMISC_THREAD_POOL_H
