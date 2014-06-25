#ifndef SIGNAL_PROCESSING_BEDROOM_H
#define SIGNAL_PROCESSING_BEDROOM_H

#include "shared_state.h"

#include <boost/exception/exception.hpp>
#include <boost/shared_ptr.hpp>

#include <set>
#include <condition_variable>

namespace Signal {
namespace Processing {

class BedroomClosed: public virtual boost::exception, public virtual std::exception {};


/**
 * @brief The Bedroom class should allow different threads to sleep on
 * this object until another thread calls wakeup().
 *
 * It should throw a BedroomClosed exception if someone tries to go to
 * sleep when the bedroom is closed.
 *
 * See Bedroom::test for usage example.
 */
class Bedroom
{
private:
    class Void {};
    typedef boost::shared_ptr<Void> Counter;

public:
    typedef std::shared_ptr<Bedroom> ptr;
    typedef std::weak_ptr<Bedroom> weak_ptr;
    class Data;

    class Bed {
    public:
        Bed(const Bed&);
        ~Bed();

        /**
         * @brief sleep sleeps indefinitely until a wakeup call. See below.
         */
        void sleep();

        /**
         * @brief sleep blocks the calling thread until Bedroom::wakeup() is called on the bedroom that created this instance.
         * @param ms_timeout time to wait for a wakeup call
         * @return true if woken up, false if the timeout elapsed before the wakeup call
         */
        bool sleep(unsigned long ms_timeout);

    private:
        friend class Bedroom;

        Bed(shared_state<Data> data);
        shared_state<Data> data_;
        Counter skip_sleep_;
    };


    class Data {
    public:
        struct shared_state_traits {
            double timeout() { return -1; }
        };

        Data();

        std::condition_variable_any work;

        std::set<Bed*> beds;
        Counter sleepers;
        Counter skip_sleep_marker;
        bool is_closed;
    };


    Bedroom();

    // Wake up sleepers
    void wakeup();
    void close();

    Bed getBed();

    int sleepers();

private:
    shared_state<Data> data_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOM_H
