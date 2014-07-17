#include "widgetoverlaycontroller.h"

#include "tools/support/toolselector.h"

// gpumisc
#include "tasktimer.h"
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
        WidgetOverlayController(QGraphicsScene* scene, RenderView* view, Tools::Commands::CommandInvoker* commandInvoker)
    : OverlayWidget(scene, view->tool_selector->parentTool()),
      pan_(0), rescale_(0), rotate_(0),
      proxy_mousepress_(0),
      view_(view),
      commandInvoker_(commandInvoker),
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
    TaskInfo(format("%s key 0x%x, modifiers 0x%x, text %s, accepted %d")
             % __FUNCTION__
             % e->key ()
             % (int)e->modifiers ()
             % e->text ().toStdString ()
             % e->isAccepted());

    if (updateFocusWidget(e))
        return;

    OverlayWidget::keyPressEvent (e);
}


void WidgetOverlayController::
        keyReleaseEvent(QKeyEvent *e)
{
    TaskInfo(format("%s key 0x%x, modifiers 0x%x, text %s, accepted %d")
             % __FUNCTION__
             % e->key ()
             % (int)e->modifiers ()
             % e->text ().toStdString ()
             % e->isAccepted());

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
    QKeyEvent key( QEvent::KeyPress, Qt::Key_unknown, event->modifiers());
    updateFocusWidget(&key);

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
    h->addWidget(pan_ = new PanWidget(view_, commandInvoker_));
    h->addWidget(rescale_ = new RescaleWidget(view_, commandInvoker_));
    h->addWidget(rotate_ = new RotateWidget(view_, commandInvoker_));
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
    v->addWidget(new PanWidget(view_, commandInvoker_));
    v->addWidget(new RescaleWidget(view_, commandInvoker_));
    v->addWidget(new RotateWidget(view_, commandInvoker_));
    v->addStretch();

    QHBoxLayout* h = new QHBoxLayout();
    h->addSpacerItem(new QSpacerItem(50,50,QSizePolicy::Maximum,QSizePolicy::Maximum));
    h->addStretch();
    h->addWidget(new PanWidget(view_, commandInvoker_));
    h->addWidget(new RescaleWidget(view_, commandInvoker_));
    h->addWidget(new RotateWidget(view_, commandInvoker_));
    h->addStretch();

    QGridLayout* g = new QGridLayout(this);
    g->addLayout(v,0,1);
    g->addLayout(h,1,0);
}


bool WidgetOverlayController::
        updateFocusWidget(QKeyEvent *e)
{
//    TaskInfo ti("%s key = 0x%x", __FUNCTION__, (int)e->key ());
//    if (((int)e->modifiers ()) & ~Qt::KeyboardModifierMask)
//        TaskInfo("modifiers = 0x%x", (int)e->modifiers ());
//    if (e->modifiers ().testFlag (Qt::MetaModifier))
//        TaskInfo("Meta");
//    if (e->modifiers ().testFlag (Qt::AltModifier))
//        TaskInfo("Alt");
//    if (e->modifiers ().testFlag (Qt::ControlModifier))
//        TaskInfo("Control");
//    if (e->modifiers ().testFlag (Qt::ShiftModifier))
//        TaskInfo("Shift");
//    if (e->modifiers ().testFlag (Qt::KeypadModifier))
//        TaskInfo("Keypad");
//    if (e->modifiers ().testFlag ((Qt::KeyboardModifier)Qt::UNICODE_ACCEL))
//        TaskInfo("UNICODE_ACCEL");
//    if (e->modifiers ().testFlag (Qt::GroupSwitchModifier))
//        TaskInfo("GroupSwitc");

    switch(e->key ())
    {
    case Qt::Key_Control:
    case Qt::Key_Meta:
    case Qt::Key_Alt:
    case Qt::Key_Shift:
    case Qt::Key_unknown:
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
        if (e->modifiers ().testFlag (Qt::ShiftModifier)) // Windows/Ubuntu Shift
            fp = pan_;
        if (e->modifiers ().testFlag (Qt::ControlModifier)) // Windows/Ubuntu Ctrl
            fp = rescale_;
        if (e->modifiers ().testFlag (Qt::AltModifier)) // Windows/Ubuntu Alt
            fp = rotate_;
#endif

        bool oldMousePress = 0!=proxy_mousepress_ && proxy_mousepress_ != fp;

        if (fp != focusProxy ())
        {
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

                    mouseReleaseEvent( &releaseEvent );
                }
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

                mousePressEvent( &pressEvent );
            }
        }

        //TaskInfo("%s", fp?vartype(*fp).c_str():0);

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
