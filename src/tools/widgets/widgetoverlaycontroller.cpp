#include "widgetoverlaycontroller.h"

// gpumisc
#include "TaskTimer.h"
#include "demangle.h"

// widgets
#include "rescalewidget.h"
#include "panwidget.h"
#include "rotatewidget.h"

// Qt
#include <QLayout>
#include <QKeyEvent>
#include <QCursor>
#include <QApplication>

#include <boost/format.hpp>

using namespace boost;

namespace Tools {
namespace Widgets {

WidgetOverlayController::
        WidgetOverlayController(RenderView* view)
    : OverlayWidget(view),
      pan_(0), rescale_(0), rotate_(0),
      proxy_mousepress_(0),
      view_(view),
      child_event_(QMouseEvent(QEvent::None,QPoint(),Qt::NoButton,0,0))
{
    setupLayout();
}


WidgetOverlayController::
        ~WidgetOverlayController()
{
}


void WidgetOverlayController::
        enterEvent ( QEvent * )
{
    setWindowOpacity (1.0);
}


void WidgetOverlayController::
        leaveEvent ( QEvent * )
{
    setWindowOpacity (0.0);
}


void WidgetOverlayController::
        keyPressEvent(QKeyEvent *e)
{
    TaskInfo("%s", str(format("%s\nkey %d\nmodifiers %d\ntext %s\naccepted %d") % __FUNCTION__ % e->key () % (int)e->modifiers () % e->text ().toStdString () % e->isAccepted()).c_str());

    if (updateFocusWidget(e))
        return;

    OverlayWidget::keyPressEvent (e);
}


void WidgetOverlayController::
        keyReleaseEvent(QKeyEvent *e)
{
    TaskInfo("%s", str(format("%s\nkey %d\nmodifiers %d\ntext %s\naccepted %d") % __FUNCTION__ % e->key () % (int)e->modifiers () % e->text ().toStdString () % e->isAccepted()).c_str());

    updateFocusWidget(e);

    OverlayWidget::keyReleaseEvent (e);
}


void WidgetOverlayController::
        mouseMoveEvent ( QMouseEvent * event )
{
    if (proxy_mousepress_)
        sendMouseProxyEvent ( event );
    else
        OverlayWidget::mouseMoveEvent( event );
}


void WidgetOverlayController::
        mousePressEvent ( QMouseEvent * event )
{
    if ((proxy_mousepress_ = focusProxy ()))
        sendMouseProxyEvent ( event );
    else
        OverlayWidget::mousePressEvent( event );
}


void WidgetOverlayController::
        mouseReleaseEvent ( QMouseEvent * event )
{
    if (proxy_mousepress_)
    {
        sendMouseProxyEvent ( event );
        proxy_mousepress_ = 0;
    }
    else
        OverlayWidget::mouseReleaseEvent( event );
}


void WidgetOverlayController::
        updatePosition()
{
    QRect r = sceneRect();
    move(r.topLeft());
    resize(r.size());
}


void WidgetOverlayController::
        setupLayout()
{
    leaveEvent (0);

    setCursor(Qt::CrossCursor);


//    setupLayoutRightAndBottom();
    setupLayoutCenter();
}


void WidgetOverlayController::
        setupLayoutCenter()
{    
    QHBoxLayout* h = new QHBoxLayout();
    h->addStretch();
    h->addWidget(pan_ = new PanWidget(view_));
    h->addWidget(rescale_ = new RescaleWidget(view_));
    h->addWidget(rotate_ = new RotateWidget(view_));
    h->addStretch();

    QVBoxLayout* v = new QVBoxLayout(this);
    v->addStretch(3);
    v->addLayout(h);
    v->addStretch(1);
}


void WidgetOverlayController::
        setupLayoutRightAndBottom()
{
    QVBoxLayout* v = new QVBoxLayout();
    v->addSpacerItem(new QSpacerItem(50,50,QSizePolicy::Maximum,QSizePolicy::Maximum));
    v->addStretch();
    v->addWidget(new PanWidget(view_));
    v->addWidget(new RescaleWidget(view_));
    v->addWidget(new RotateWidget(view_));
    v->addStretch();

    QHBoxLayout* h = new QHBoxLayout();
    h->addSpacerItem(new QSpacerItem(50,50,QSizePolicy::Maximum,QSizePolicy::Maximum));
    h->addStretch();
    h->addWidget(new PanWidget(view_));
    h->addWidget(new RescaleWidget(view_));
    h->addWidget(new RotateWidget(view_));
    h->addStretch();

    QGridLayout* g = new QGridLayout(this);
    g->addLayout(v,0,1);
    g->addLayout(h,1,0);
}


bool WidgetOverlayController::
        updateFocusWidget(QKeyEvent *e)
{
    switch(e->key ())
    {
    case Qt::Key_Control:
    case Qt::Key_Meta:
    case Qt::Key_Alt:
    case Qt::Key_Shift:
    {
        QWidget* fp = 0;
#ifdef __APPLE__
        if (e->modifiers ().testFlag (Qt::MetaModifier)) // Mac Ctrl
            fp = pan_;
        if (e->modifiers ().testFlag (Qt::AltModifier)) // Mac Alt
            fp = rescale_;
        if (e->modifiers ().testFlag (Qt::ControlModifier)) // Mac Cmd
            fp = rotate_;
#else
        if (e->modifiers ().testFlag (Qt::ShiftModifier)) // Windows Shift
            fp = pan_;
        if (e->modifiers ().testFlag (Qt::ControlModifier)) // Windows Ctrl
            fp = rescale_;
        if (e->modifiers ().testFlag (Qt::AltModifier)) // Windows Alt
            fp = rotate_;
#endif

        if (fp != focusProxy ())
        {
            bool oldMousePress = 0!=proxy_mousepress_  && proxy_mousepress_ != fp;

            if (fp)
            {
                fp->setFocus ( Qt::MouseFocusReason );
                QApplication::setOverrideCursor (fp->cursor ());

                if (oldMousePress)
                {
                    QMouseEvent releaseEvent(
                                QEvent::MouseButtonRelease,
                                lastMousePos_,
                                child_event_.button (),
                                child_event_.buttons (),
                                child_event_.modifiers ()
                                );
                    mouseReleaseEvent(&releaseEvent);
                }
                else
                    proxy_mousepress_ = fp;
            }

            setFocusProxy (fp);

            if (!fp)
            {
                setFocus();
                QApplication::restoreOverrideCursor ();
            }
            else if (oldMousePress)
            {
                QMouseEvent pressEvent(
                            QEvent::MouseButtonPress,
                            lastMousePos_,
                            child_event_.button (),
                            child_event_.buttons (),
                            child_event_.modifiers ()
                            );
                mousePressEvent(&pressEvent);
            }
        }
        return true;
    }
    default:
        return false;
    }
}


void WidgetOverlayController::
        sendMouseProxyEvent( QMouseEvent * e )
{
    lastMousePos_ = e->pos();
    child_event_ = QMouseEvent(
                e->type (),
                proxy_mousepress_->mapFromParent (e->pos ()),
                e->button (),
                e->buttons (),
                e->modifiers ()
                );

    QCoreApplication::sendEvent (proxy_mousepress_, &child_event_);
}


} // namespace Widgets
} // namespace Tools
