#ifndef UI_UPDATEWIDGETSINK_H
#define UI_UPDATEWIDGETSINK_H

#include "signal/sink.h"
#include <QObject>

namespace Ui {

class UpdateWidgetSink: public QObject, public Signal::Sink
{
    Q_OBJECT
public:
    UpdateWidgetSink(QWidget* w);

    virtual void put(Signal::pBuffer);

signals:
    void update();
};

} // namespace Ui

#endif // UI_UPDATEWIDGETSINK_H
