#ifndef SIGNAL_PROCESSING_INOTIFIER_H
#define SIGNAL_PROCESSING_INOTIFIER_H

#include <boost/shared_ptr.hpp>

namespace Signal {
namespace Processing {

class INotifier
{
public:
    typedef boost::weak_ptr<INotifier> WeakPtr;
    typedef boost::shared_ptr<INotifier> Ptr;

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
