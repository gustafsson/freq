#ifndef SIGNAL_PROCESSING_BEDROOM_H
#define SIGNAL_PROCESSING_BEDROOM_H

#include <QSemaphore>

#include "volatileptr.h"

namespace Signal {
namespace Processing {

class Void {};
typedef boost::shared_ptr<Void> Bed;

/**
 * @brief The Bedroom class should allow different threads to sleep on
 * this object until another thread calls wakeup().
 */
class Bedroom: public VolatilePtr<Bedroom>
{
public:
    Bedroom();

    // Wake up sleepers
    int wakeup() volatile;

    void sleep() volatile;
    void sleep(int ms_timeout) volatile;

    int sleepers() const volatile;

private:
    QSemaphore work_;
    Bed bed_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOM_H
