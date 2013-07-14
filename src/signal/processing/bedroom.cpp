#include "bedroom.h"

#include "exceptionassert.h"

namespace Signal {
namespace Processing {

void Bedroom::
        wakeup() volatile
{
    WritePtr(this)->work_condition.wakeAll ();
}


void Bedroom::
        sleep() volatile
{
    // QWaitCondition/QMutex are thread-safe so we can discard the volatile qualifier
    const_cast<QWaitCondition*>(&work_condition)->wait (
                const_cast<QMutex*> (&work_condition_mutex));
}

} // namespace Processing
} // namespace Signal


#include <QThread>

namespace Signal {
namespace Processing {

class SleepyFaceMock: public QThread {
public:
    SleepyFaceMock(Bedroom::Ptr bedroom, int snooze) : bedroom_(bedroom), snooze_(snooze) {}

    void run() {
        do {
            bedroom_->sleep();
        } while(--snooze_ > 0);
    }

    int snooze() { return snooze_; }
private:
    Bedroom::Ptr bedroom_;
    int snooze_;
};


void Bedroom::
        test()
{
    // It should allow different threads to sleep on this object until another thread calls wakeup()
    {
        Bedroom::Ptr bedroom(new Bedroom);
        int snoozes = 10;
        SleepyFaceMock sleepyface1(bedroom, snoozes);
        SleepyFaceMock sleepyface2(bedroom, snoozes);

        sleepyface1.start ();
        sleepyface2.start ();

        for (int i=snoozes; i>=0; i--) {
            usleep(1000);
            bedroom->wakeup();
            EXCEPTION_ASSERT_EQUALS(sleepyface1.snooze (), i);
            EXCEPTION_ASSERT_EQUALS(sleepyface2.snooze (), i);
        }

        EXCEPTION_ASSERT(sleepyface1.isFinished ());
        EXCEPTION_ASSERT(sleepyface2.isFinished ());
    }
}

} // namespace Processing
} // namespace Signal
