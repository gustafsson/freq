#include "callslotevent.h"
#include "TaskTimer.h"
#include "demangle.h"

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
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

    QTimer::singleShot(1, receiver, slotname);
}


QTestEvent *CallSlotEvent::
        clone() const
{
    TaskTimer ti("%s::%s", vartype(*this).c_str(), __FUNCTION__, NULL);

    return new CallSlotEvent(receiver, slotname);
}
