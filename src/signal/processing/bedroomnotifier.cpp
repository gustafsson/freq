#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

BedroomNotifier::
        BedroomNotifier(Bedroom::Ptr bedroom)
    :
    bedroom_(bedroom)
{
}


void BedroomNotifier::
        wakeup() const
{
    Bedroom::Ptr bedroom = bedroom_.lock();
    if (bedroom)
        bedroom->wakeup();
}

} // namespace Processing
} // namespace Signal
