#ifndef CALLSLOTEVENT_H
#define CALLSLOTEVENT_H

#include <QTestEvent>

class CallSlotEvent : public QTestEvent
{
public:
    CallSlotEvent(QObject* receiver, const char* slotname);

    virtual void simulate(QWidget *);

    virtual QTestEvent *clone() const;

private:
    QObject* receiver;
    const char* slotname;
};

#endif // CALLSLOTEVENT_H
