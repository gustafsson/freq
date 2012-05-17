#include "updatewidgetsink.h"
#include <QWidget>

namespace Ui {

UpdateWidgetSink::
        UpdateWidgetSink(QWidget* w)
{
    connect( this, SIGNAL(update()), w, SLOT(update()), Qt::AutoConnection);
}


void UpdateWidgetSink::
        put(Signal::pBuffer)
{
    // Instead of calling w->update() directly, emit a signal that will put
    // the message in a queue if put is called from a different thread, which
    // it is for instance when a microphonerecorder calls it.
    emit update();
}

} // namespace Ui
