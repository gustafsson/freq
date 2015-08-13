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

    blocking_queue(){}
    blocking_queue(const blocking_queue&)=delete;
    blocking_queue& operator=(const blocking_queue&)=delete;
    blocking_queue(blocking_queue&&)=default;
    blocking_queue& operator=(blocking_queue&&)=default;

    class abort_exception : public std::exception {};

    ~blocking_queue() {
        close ();
    }

    queue clear()
    {
        std::unique_lock<std::mutex> l(m);
        queue p;
        p.swap (q);
        return p;
    }

    // this method is used to abort any blocking pop,
    // clear the queue, disable any future pops and
    // discard any future pushes
    void close() {
        std::unique_lock<std::mutex> l(m);
        abort_ = true;
        queue().swap (q);
        l.unlock ();
        c.notify_all ();
    }

    bool empty() {
        std::unique_lock<std::mutex> l(m);
        return q.empty ();
    }

    T pop() {
        std::unique_lock<std::mutex> l(m);

        c.wait (l, [this](){return !q.empty() || abort_;});

        if (abort_)
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

        if (!c.wait_for (l, d, [this](){return !q.empty() || abort_;}))
            return T();

        if (abort_)
            throw abort_exception{};

        T t( std::move(q.front()) );
        q.pop ();
        return t;
    }

    /**
     * @brief pop0 does not block but returns T() if the queue is empty.
     * @return
     */
    T pop0() {
        std::unique_lock<std::mutex> l(m);

        if (q.empty ())
            return T();

        T t( std::move(q.front()) );
        q.pop ();
        return t;
    }

    void push(const T& t) {
        std::unique_lock<std::mutex> l(m);
        if (!abort_)
            q.push (t);
        l.unlock ();
        c.notify_one ();
    }

    void push(T&& t) {
        std::unique_lock<std::mutex> l(m);
        if (!abort_)
            q.push (std::move(t));
        l.unlock ();
        c.notify_one ();
    }

    /**
     * Waits for the queue to become empty and returns the size of the queue.
     */
    template <class Rep, class Period>
    int wait_for(const std::chrono::duration<Rep, Period>& d) {
        std::unique_lock<std::mutex> l(m);

        c.wait_for (l, d, [this](){return q.empty();});

        return q.size ();
    }

    /**
     * Waits for the queue to become empty and returns the size of the queue.
     */
    int wait() {
        std::unique_lock<std::mutex> l(m);

        c.wait (l, [this](){return q.empty();});

        return q.size ();
    }
private:
    bool abort_ = false;
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
