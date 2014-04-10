#ifndef JUSTMISC_BLOCKING_QUEUE_H
#define JUSTMISC_BLOCKING_QUEUE_H

#include <mutex>
#include <queue>

namespace JustMisc {

/**
 * @brief The blocking_queue class should provide a thread safe solution to the
 * multiple consumer-multiple producer pattern.
 */
template<class T>
class blocking_queue
{
public:
    typedef T value_type;
    typedef std::queue<T> queue;

    class abort_exception : public std::exception {};

    ~blocking_queue() {
        abort_on_empty ();
        clear ();
    }

    queue clear()
    {
        std::unique_lock<std::mutex> l(m);
        std::queue<T> q;
        q.swap (this->q);
        return q;
    }

    void abort_on_empty() {
        std::unique_lock<std::mutex> l(m);
        abort_on_empty_ = true;
        l.unlock ();
        c.notify_all ();
    }

    bool empty() {
        std::unique_lock<std::mutex> l(m);
        return q.empty ();
    }

    T pop() {
        std::unique_lock<std::mutex> l(m);

        c.wait (l, [this](){return !q.empty() || abort_on_empty_;});

        if (abort_on_empty_)
            throw abort_exception{};

        T t( std::move(q.front()) );
        q.pop ();
        return t;
    }

    /**
     * Returns default constructed T on timeout.
     */
    template <class Rep, class Period>
    T pop_for(const std::chrono::duration<Rep, Period>& d) {
        std::unique_lock<std::mutex> l(m);

        if (!c.wait_for (l, d, [this](){return !q.empty() || abort_on_empty_;}))
            return T();

        if (abort_on_empty_)
            throw abort_exception{};

        T t( std::move(q.front()) );
        q.pop ();
        return t;
    }

    void push(const T& t) {
        std::unique_lock<std::mutex> l(m);
        q.push (t);
        l.unlock ();
        c.notify_one ();
    }

    void push(T&& t) {
        std::unique_lock<std::mutex> l(m);
        q.push (std::move(t));
        l.unlock ();
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

} // namespace JustMisc

#endif // JUSTMISC_BLOCKING_QUEUE_H
