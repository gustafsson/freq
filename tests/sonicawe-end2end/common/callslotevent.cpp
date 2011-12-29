#include "callslotevent.h"

#include <QTimer>

CallSlotEvent::
        CallSlotEvent(QObject* receiver, const char* slotname)
            :
            receiver(receiver),
            slotname(slotname)
{}


void CallSlotEvent::
        simulate(QWidget *)
{
    QTimer::singleShot(1, receiver, slotname);
}

QTestEvent *CallSlotEvent::
        clone() const
{
    return new CallSlotEvent(receiver, slotname);
}
