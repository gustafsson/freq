#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

BedroomNotifier::
        BedroomNotifier(Bedroom::ptr bedroom)
    :
    bedroom_(bedroom)
{
}


void BedroomNotifier::
        wakeup() const
{
    Bedroom::ptr bedroom = bedroom_.lock();
    if (bedroom)
        bedroom->wakeup();
}

} // namespace Processing
} // namespace Signal
