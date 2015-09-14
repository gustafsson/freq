#ifndef SIGNAL_PROCESSING_INOTIFIER_H
#define SIGNAL_PROCESSING_INOTIFIER_H

#include <memory>

namespace Signal {
namespace Processing {

class INotifier
{
public:
    typedef std::shared_ptr<INotifier> ptr;
    typedef std::weak_ptr<INotifier> weak_ptr;

    virtual ~INotifier() {}

    /**
     * @brief wakeup
     * Reentrant.
     */
    virtual void wakeup() const=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_INOTIFIER_H
