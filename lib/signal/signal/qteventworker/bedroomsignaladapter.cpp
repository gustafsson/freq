#include "bedroomsignaladapter.h"

#include "tasktimer.h"

namespace Signal {
namespace QtEventWorker {

//#define DEBUGINFO
#define DEBUGINFO if(0)

BedroomSignalAdapter::
        BedroomSignalAdapter(Processing::Bedroom::ptr bedroom, QObject* parent)
    :
    QThread(parent),
    bedroom_(bedroom),
    stop_flag_(false)
{
    start();
}


BedroomSignalAdapter::
        ~BedroomSignalAdapter()
{
    quit_and_wait();
}


void BedroomSignalAdapter::
        quit_and_wait ()
{
    DEBUGINFO TaskTimer ti("BedroomSignalAdapter quit_and_wait");
    stop_flag_ = true;

    bedroom_->wakeup();

    QThread::wait ();
}


void BedroomSignalAdapter::
        run ()
{
    auto bed = bedroom_->getBed();
    while (!stop_flag_) {
        DEBUGINFO TaskInfo("BedroomSignalAdapter wakeup");
        emit wakeup();

        try {
            bed.sleep ();
        } catch (const Processing::BedroomClosed&) {
            stop_flag_ = true;
        }
    }

    DEBUGINFO TaskInfo("BedroomSignalAdapter finished");
}

} // namespace Processing
} // namespace Signal
