#include "bedroom.h"

#include "exceptionassert.h"

namespace Signal {
namespace Processing {


Bedroom::
        Bedroom()
    :
      bed_(new Bed::value_type)
{
}


int Bedroom::
        wakeup() volatile
{
    int N = sleepers();
    int should_be_available = std::max(1, N);

    WritePtr self(this);

    int to_release = should_be_available - self->work_.available ();
    if (0<to_release)
        self->work_.release (to_release);

    return std::max(0,to_release);
}


void Bedroom::
        sleep() volatile
{
    sleep(-1);
}


void Bedroom::
        sleep(int ms_timeout) volatile
{
    // Increment usage count
    Bed b = ReadPtr(this)->bed_;

    // Don't keep a lock when acquiring
    const_cast<Bedroom*>(this)->work_.tryAcquire (1, ms_timeout);
}


int Bedroom::
        sleepers() const volatile
{
    // Remove 1 to compensate for the instance used by 'this'
    return ReadPtr(this)->bed_.use_count() - 1;
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
            --snooze_;
            usleep(60);
        } while(snooze_ > 0);
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
    for (int j=0;j<2; j++) {
        Bedroom::Ptr bedroom(new Bedroom);
        int snoozes = 10;
        SleepyFaceMock sleepyface1(bedroom, snoozes);
        SleepyFaceMock sleepyface2(bedroom, snoozes);

        sleepyface1.start ();
        sleepyface2.start ();

        for (int i=snoozes; i>=0; i--) {
            EXCEPTION_ASSERT_EQUALS(sleepyface1.wait (2), i>0?false:true);
            EXCEPTION_ASSERT_EQUALS(sleepyface2.wait (2), i>0?false:true);

            // sleepyface1 and sleepyface2 shoule be sleeping now
            EXCEPTION_ASSERT_EQUALS(bedroom->sleepers(), i>0?2:0);

            // they should have 'i' times left to snooze
            EXCEPTION_ASSERTX(sleepyface1.snooze () == i && sleepyface2.snooze () == i,
                              (boost::format("sleepyface1=%d, sleepyface2=%d, i=%d")
                              % sleepyface1.snooze () % sleepyface2.snooze () % i));

            // TODO Bedroom::wakeup may wake the same sleepyface twice instead of waking up both,
            // this behaviour is valid but it doesn't pass this test.
            int w = bedroom->wakeup();
            EXCEPTION_ASSERT_EQUALS(w, i>0?2:1);
        }

        EXCEPTION_ASSERT(sleepyface1.isFinished ());
        EXCEPTION_ASSERT(sleepyface2.isFinished ());
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers(), 0);
    }
}

} // namespace Processing
} // namespace Signal
