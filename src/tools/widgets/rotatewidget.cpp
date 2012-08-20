#include "rotatewidget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QApplication>

#include "hudglwidget.h"
#include "sawe/project.h"
#include "tools/renderview.h"
#include "tools/commands/rotatecameracommand.h"

namespace Tools {
namespace Widgets {


RotateWidget::RotateWidget(RenderView* view) :
    view_(view),
    mouseMoved_(true)
{
    setMinimumSize(50,50);
    setCursor(Qt::OpenHandCursor);
    setToolTip("Click to flip between 2D and 3D. Drag to rotate in 3D");
}


void RotateWidget::
        leaveEvent ( QEvent * )
{
    QApplication::restoreOverrideCursor ();
}


void RotateWidget::
        mouseMoveEvent ( QMouseEvent * event )
{
    QPoint d = event->pos() - dragSource_;
    float dt = d.x();
    float ds = d.y();

    Tools::Commands::pCommand cmd( new Tools::Commands::RotateCameraCommand(view_->model, dt, ds ));
    view_->model->project()->commandInvoker()->invokeCommand( cmd );

    dragSource_ = event->pos();
    mouseMoved_ = true;
}


void RotateWidget::
        mousePressEvent ( QMouseEvent * event )
{
    QApplication::setOverrideCursor (Qt::ClosedHandCursor);
    dragSource_ = event->pos();
    mouseMoved_ = false;
}


void RotateWidget::
        mouseReleaseEvent ( QMouseEvent * event )
{
    if (!mouseMoved_)
    {
        float rx = view_->model->_rx;
        bool is2D = rx >= 90;
//        float rx = view_->model->_rx;
        float tx3D = 45;
        float tx2D = 91;
        float d = is2D ? tx3D - rx : tx2D - rx;

        const float rs = 0.2;
        Tools::Commands::pCommand cmd( new Tools::Commands::RotateCameraCommand(view_->model, 0, d/rs ));
        view_->model->project()->commandInvoker()->invokeCommand( cmd );
        view_->model->orthoview.TimeStep(1);
    }

    leaveEvent(event);
}


void RotateWidget::
        paintEvent ( QPaintEvent * event )
{
    QPainter painter (this);
    painter.beginNativePainting ();
    painter.setRenderHints (QPainter::Antialiasing | QPainter::HighQualityAntialiasing);
    //painter.fillPath (path_, QColor(125,125,125,125));
    painter.fillPath (path_, QColor(220,220,220,200));
    painter.strokePath (path_, QPen(
                            QColor(100,100,100,200),
                            .8,
                            Qt::SolidLine,
                            Qt::RoundCap,
                            Qt::RoundJoin));

    QWidget::paintEvent (event);
}



void RotateWidget::
        resizeEvent ( QResizeEvent *)
{
    recreatePolygon();
}


void RotateWidget::
        recreatePolygon ()
{
    path_ = QPainterPath();
    //path_.addPolygon(circleShape());
    path_.addPolygon(bunkShape());

    update();
}


QPolygon RotateWidget::
        circleShape ()
{
    QPoint o = rect().center();
    QPoint x = (rect().topRight() - rect().topLeft())/4;
    QPoint y = (rect().topLeft() - rect().bottomLeft())/4;

    QPolygon poly;
    float N = std::max(width(), height());
    for (float i=0; i<=N; ++i)
    {
        float a = -2*M_PI*(i/N) * 4/5;
        float r = 1.0;
        poly.push_back(o + r*x*cos(a) + r*y*sin(a));
    }

    poly.push_back( o + .9*y + .4*x );
    setMask(HudGlWidget::growRegion(poly));

    for (float i=N; i>=0; --i)
    {
        float a = -2*M_PI*(i/N) * 4/5;
        float r = 0.65;
        poly.push_back(o + r*x*cos(a) + r*y*sin(a));
    }

    return poly;
}


QPolygon RotateWidget::
        bunkShape ()
{
    QPoint o = rect().center();
    QPoint x = (rect().topRight() - rect().topLeft())/4;
    QPoint y = (rect().topLeft() - rect().bottomLeft())/4;

    o = o + 0.4*x - 0.55*y;

    QPolygon poly;
    poly.push_back(o);
    float N = std::max(width(), height());
    for (float i=0; i<=N; ++i)
    {
        float a = 2*M_PI*(i/N)*6/20;
        float r = 1.25;
        poly.push_back(o - r*x*cos(a) + r*y*sin(a));
    }

    setMask(HudGlWidget::growRegion(poly));

    return poly;
}


} // namespace Widgets
} // namespace Tools
