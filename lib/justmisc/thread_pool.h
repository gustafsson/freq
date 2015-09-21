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
    thread_pool(const char* name=0);
    thread_pool(int N, const char* name=0);
    thread_pool(const thread_pool&)=delete;
    thread_pool& operator=(const thread_pool&)=delete;
    thread_pool(thread_pool&&)=default;
    thread_pool& operator=(thread_pool&&)=default;
    ~thread_pool();

    void addTask(std::packaged_task<void()>&& task)
    {
        queue_.push (std::move(task));
    }

    // The return value may be fetched by a future.
    template<class F>
    void addTask(std::packaged_task<F()>&& task)
    {
        queue_.push (std::packaged_task<void()>(
                [task=std::move(task)] () mutable
                {
                    task();
                }
        ));
    }

    size_t thread_count () const { return threads_.size (); }

    /**
     * Waits for the queue to become empty and returns the size of the queue.
     */
    template <class Rep, class Period>
    int wait_for(const std::chrono::duration<Rep, Period>& d) {
        return queue_.wait_for(d);
    }

    /**
     * Waits for the queue to become empty and returns the size of the queue.
     */
    int wait() {
        return queue_.wait();
    }

private:
    blocking_queue<std::packaged_task<void()>> queue_;
    std::vector<std::thread> threads_;

public:
    static void test();
};

} // namespace JustMisc

#endif // JUSTMISC_THREAD_POOL_H
