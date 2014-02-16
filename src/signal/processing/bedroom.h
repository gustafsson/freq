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
class Bedroom: public VolatilePtr<Bedroom>
{
public:
    class Bed;

    class Void {};
    typedef boost::shared_ptr<Void> Counter;

    class Data: public VolatilePtr<Data> {
    public:
        Data();

        boost::condition_variable_any work;

        std::set<Bed*> beds;
        Counter sleepers;
        Counter skip_sleep_marker;
        bool is_closed;

        VolatilePtr<Data>::shared_mutex* readWriteLock() const volatile;
    };


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

        Bed(Data::Ptr data);
        Data::Ptr data_;
        Counter skip_sleep_;
    };


    Bedroom();

    // Wake up sleepers
    void wakeup() volatile;
    void close() volatile;

    Bed getBed() volatile;

    int sleepers() volatile;

private:
    Data::Ptr data_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOM_H
