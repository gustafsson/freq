#include "bedroom.h"

#include "exceptionassert.h"
#include "expectexception.h"

#include <QSemaphore>

namespace Signal {
namespace Processing {


class Void {};
typedef boost::shared_ptr<Void> Bed;


class BedroomData {
public:
    BedroomData()
        :
          bed(new Bed::value_type),
          is_closed(false)
    {
    }

    // TODO use QWaitCondition
    QSemaphore work;
    Bed bed;
    bool is_closed;
};


Bedroom::
        Bedroom(Bedroom::DataPtr data)
    :
      data_(data)
{
    if (!data_)
        data_.reset (new BedroomData);
}


int Bedroom::
        wakeup() volatile
{
    int N = sleepers();
    int should_be_available = std::max(1, N);

    WritePtr self(this);

    int to_release = should_be_available - self->data_->work.available ();
    if (0<to_release)
        self->data_->work.release (to_release);

    return std::max(0,to_release);
}


void Bedroom::
        close() volatile
{
    WritePtr(this)->data_->is_closed = true;
    wakeup();
}


void Bedroom::
        sleep() volatile
{
    sleep(-1);
}


void Bedroom::
        sleep(int ms_timeout) volatile
{
    if (ReadPtr(this)->data_->is_closed)
    {
        BOOST_THROW_EXCEPTION(BedroomClosed() << Backtrace::make ());
    }

    // Increment usage count
    Bed b = ReadPtr(this)->data_->bed;

    // Don't keep a lock to this when acquiring
    const_cast<Bedroom*>(this)->data_->work.tryAcquire (1, ms_timeout);
}


int Bedroom::
        sleepers() const volatile
{
    // Remove 1 to compensate for the instance used by 'this'
    return ReadPtr(this)->data_->bed.use_count() - 1;
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

    // It should throw a BedroomClosed exception if someone tries to go to
    // sleep when the bedroom is closed.
    {
        Bedroom b;
        b.close ();
        EXPECT_EXCEPTION(BedroomClosed, b.sleep ());
    }
}

} // namespace Processing
} // namespace Signal
