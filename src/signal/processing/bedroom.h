#ifndef SIGNAL_PROCESSING_BEDROOM_H
#define SIGNAL_PROCESSING_BEDROOM_H

#include "volatileptr.h"

namespace Signal {
namespace Processing {

class BedroomClosed: public virtual boost::exception, public virtual std::exception {};


/**
 * @brief The Bedroom class should allow different threads to sleep on
 * this object until another thread calls wakeup().
 *
 * It should throw a BedroomClosed exception if someone tries to go to
 * sleep when the bedroom is closed.
 */
class Bedroom: public VolatilePtr<Bedroom>
{
public:
    typedef boost::shared_ptr<class BedroomData> DataPtr;

    Bedroom(DataPtr d=DataPtr());

    // Wake up sleepers
    int wakeup() volatile;
    void close() volatile;

    void sleep() volatile;
    void sleep(int ms_timeout) volatile;

    int sleepers() const volatile;


private:
    DataPtr data_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOM_H
