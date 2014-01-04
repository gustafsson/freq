#include "bedroomsignaladapter.h"

namespace Signal {
namespace Processing {

BedroomSignalAdapter::
        BedroomSignalAdapter(Bedroom::Ptr bedroom, QObject* parent)
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
    stop_flag_ = true;

    bedroom_->wakeup();

    QThread::wait ();
}


void BedroomSignalAdapter::
        run ()
{
    Bedroom::Bed bed = bedroom_->getBed();
    while (!stop_flag_) {
        emit wakeup();

        try {
            bed.sleep ();
        } catch (const BedroomClosed&) {
            stop_flag_ = true;
        }
    }
}

} // namespace Processing
} // namespace Signal
