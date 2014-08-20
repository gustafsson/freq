#ifndef SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H
#define SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H

#include <QtCore>
#include "bedroom.h"
#include "inotifier.h"

namespace Signal {
namespace Processing {

/**
 * @brief The BedroomSignalAdapter class should translate a wakeup signal from
 * a bedroom to a Qt Signal.
 */
class BedroomSignalAdapter : public QThread
{
    Q_OBJECT
public:
    explicit BedroomSignalAdapter(Bedroom::ptr bedroom, QObject* parent);
    ~BedroomSignalAdapter();

    /**
     * @brief quit_and_wait will prevent any more 'wakeup_signal's from being
     * emitted and finish this thread.
     */
    void quit_and_wait ();

signals:
    void wakeup();

private:
    // QThread
    void run ();

    Bedroom::ptr bedroom_;
    bool stop_flag_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H
