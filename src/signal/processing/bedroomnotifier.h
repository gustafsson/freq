#ifndef SIGNAL_PROCESSING_BEDROOMNOTIFIER_H
#define SIGNAL_PROCESSING_BEDROOMNOTIFIER_H

#include <QObject>
#include "bedroom.h"
#include "inotifier.h"

namespace Signal {
namespace Processing {

class BedroomNotifier : public QObject, public INotifier
{
    Q_OBJECT
public:
    explicit BedroomNotifier(Bedroom::Ptr bedroom);

    void wakeup() const;

signals:

public slots:

private:
    Bedroom::WeakPtr bedroom_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_BEDROOMNOTIFIER_H
