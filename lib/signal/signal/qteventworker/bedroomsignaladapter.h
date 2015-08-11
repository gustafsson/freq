#ifndef SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H
#define SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H

#include <QtCore>
#include "signal/processing/bedroom.h"
#include "signal/processing/inotifier.h"

namespace Signal {
namespace QtEventWorker {

/**
 * @brief The BedroomSignalAdapter class should translate a wakeup signal from
 * a bedroom to a Qt Signal.
 */
class BedroomSignalAdapter : public QThread
{
    Q_OBJECT
public:
    explicit BedroomSignalAdapter(Processing::Bedroom::ptr bedroom, QObject* parent);
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

    Processing::Bedroom::ptr bedroom_;
    bool stop_flag_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOMSIGNALADAPTER_H
