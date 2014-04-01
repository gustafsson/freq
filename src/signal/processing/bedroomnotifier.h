#ifndef SIGNAL_PROCESSING_BEDROOMNOTIFIER_H
#define SIGNAL_PROCESSING_BEDROOMNOTIFIER_H

#include "bedroom.h"
#include "inotifier.h"

namespace Signal {
namespace Processing {

class BedroomNotifier : public INotifier
{
public:
    explicit BedroomNotifier(Bedroom::Ptr bedroom);

    void wakeup() const;

private:
    Bedroom::weak_ptr bedroom_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOMNOTIFIER_H
