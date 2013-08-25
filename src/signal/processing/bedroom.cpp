#include "bedroom.h"

#include "exceptionassert.h"
#include "expectexception.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {


Bedroom::Data::
    Data()
    :
        sleepers(new Counter::value_type),
        skip_sleep_marker(new Counter::value_type),
        is_closed(false)
{
}


QReadWriteLock* Bedroom::Data::
        readWriteLock() const volatile
{
    return VolatilePtr<Data>::readWriteLock();
}


Bedroom::Bed::
        Bed(Data::Ptr data)
    :
      data_(data)
{
    write1(data_)->beds.insert(this);
}


Bedroom::Bed::
        ~Bed()
{
    write1(data_)->beds.erase(this);
}


void Bedroom::Bed::
        sleep()
{
    sleep(ULONG_MAX);
}


void Bedroom::Bed::
        sleep(unsigned long ms_timeout)
{
    Data::WritePtr data(data_);

    if (data->is_closed) {
        BOOST_THROW_EXCEPTION(BedroomClosed() << Backtrace::make ());
    }

    if (!skip_sleep_) {
        // Increment usage count
        Counter b = data->sleepers;

        data->work.wait (data_->readWriteLock(), ms_timeout);
    }

    skip_sleep_.reset();

    return;
}


Bedroom::
        Bedroom()
    :
      data_(new Data)
{
}


void Bedroom::
        wakeup() volatile
{
    Data::WritePtr data(WritePtr(this)->data_);
    // no one is going into sleep as long as data_ is locked

    BOOST_FOREACH(Bed* b, data->beds) {
        b->skip_sleep_ = data->skip_sleep_marker;
    }

    data->work.wakeAll ();
}


void Bedroom::
        close() volatile
{
    {
        Data::WritePtr data(WritePtr(this)->data_);
        data->is_closed = true;
    }

    wakeup();
}


Bedroom::Bed Bedroom::
        getBed() volatile
{
    return Bed(ReadPtr(this)->data_);
}


int Bedroom::
        sleepers() volatile
{
    // Remove 1 to compensate for the instance used by 'this'
    Data::ReadPtr data(ReadPtr(this)->data_);
    return data->sleepers.use_count() - 1;
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
        Bedroom::Bed bed = bedroom_->getBed();
        do {
            bed.sleep ();
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

            bedroom->wakeup();
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
        EXPECT_EXCEPTION(BedroomClosed, b.getBed().sleep ());
    }
}

} // namespace Processing
} // namespace Signal
