#ifndef SIGNAL_PROCESSING_BEDROOM_H
#define SIGNAL_PROCESSING_BEDROOM_H

#include "volatileptr.h"

#include <set>

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
    friend class Bed;
    class Data;
    class Void {};
    typedef boost::shared_ptr<Void> Counter;

public:
    typedef std::shared_ptr<Bedroom> Ptr;
    typedef std::weak_ptr<Bedroom> WeakPtr;

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

        Bed(VolatilePtr<Data> data);
        VolatilePtr<Data> data_;
        Counter skip_sleep_;
    };


    Bedroom();

    // Wake up sleepers
    void wakeup();
    void close();

    Bed getBed();

    int sleepers();

private:
    class Data {
    public:
        typedef VolatilePtr<Data> Ptr;
        typedef Ptr::WritePtr WritePtr;

        struct VolatilePtrTypeTraits {
            int timeout_ms() { return -1; }
            int verify_execution_time_ms() { return -1; }
            VerifyExecutionTime::report report_func() { return 0; }
        };

        Data();

        boost::condition_variable_any work;

        std::set<Bed*> beds;
        Counter sleepers;
        Counter skip_sleep_marker;
        bool is_closed;
    };

    Data::Ptr data_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

template<>
class VolatilePtrTypeTraits<Signal::Processing::Bedroom::Data> {
public:
    int timeout_ms() { return -1; }
    int verify_execution_time_ms() { return -1; }
    VerifyExecutionTime::report report_func() { return 0; }
};

#endif // SIGNAL_PROCESSING_BEDROOM_H
