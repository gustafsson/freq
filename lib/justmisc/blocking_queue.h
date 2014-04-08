#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <mutex>
#include <queue>

/**
 * @brief The blocking_queue class should provide a thread safe solution to the
 * multiple consumer-multiple producer pattern.
 */
template<class T>
class blocking_queue
{
public:
    class abort_exception : public std::exception {};

    void clear() {
        std::unique_lock<std::mutex> l(m);
        while (!q.empty ())
            q.pop ();
    }

    void abort_on_empty() {
        abort_on_empty_ = true;
        c.notify_all ();
    }

    T pop() {
        std::unique_lock<std::mutex> l(m);
        while (q.empty())
        {
            if (abort_on_empty_)
                throw abort_exception{};
            c.wait (l);
        }

        T t( std::move(q.front ()) );
        q.pop ();
        return t;
    }

    void push(T t) {
        std::unique_lock<std::mutex> l(m);
        q.push (std::move(t));
        c.notify_one ();
    }

private:
    bool abort_on_empty_ = false;
    std::queue<T> q;
    std::mutex m;
    std::condition_variable c;
};

class blocking_queue_test {
public:
    static void test();
};

#endif // BLOCKING_QUEUE_H
