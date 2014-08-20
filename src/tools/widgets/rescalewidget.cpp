#include "rescalewidget.h"

#include <QtWidgets> // QApplication

#include "sawe/project.h"
#include "tools/renderview.h"
#include "tools/commands/zoomcameracommand.h"
#include "gl.h"

const bool DIRECT_RESCALING = true;

using namespace std;

namespace Tools {
namespace Widgets {

RescaleWidget::
        RescaleWidget (RenderView*v)
    :   HudGlWidget(v),
        view_(v),
        scalex_(1.f),
        scaley_(1.f),
        image_(":/icons/muchdifferent.png"),
        qimage_(":/icons/muchdifferent.png"),
        updateTimer_(0)
{
    setMinimumSize(70,70);
    setCursor(Qt::SizeAllCursor);
#ifdef __APPLE__
    setToolTip("Click and drag to rescale axes [hold alt]");
#else
    setToolTip("Click and drag to rescale axes [hold ctrl]");
#endif

    //this->setMinimumSize(qimage_.size());
    //QRegion m(QBitmap(qimage_.size()).fromImage(qimage_.createAlphaMask()));
    //setMask( growRegion(m,2) );
}


void RescaleWidget::
        timerEvent( QTimerEvent *)
{
    updateModel();
}


void RescaleWidget::
        leaveEvent ( QEvent * )
{
    scalex_ = 1.f;
    scaley_ = 1.f;
    recreatePolygon();
    killTimer(updateTimer_);
}


void RescaleWidget::
        mouseMoveEvent ( QMouseEvent * event )
{
    lastPos_ = event->pos();

    if (DIRECT_RESCALING)
    {
        updateModel();
        dragSource_ = lastPos_;
    }
}


void RescaleWidget::
        mousePressEvent ( QMouseEvent * event )
{
    lastPos_ = dragSource_ = event->pos();

    if (!DIRECT_RESCALING)
        updateTimer_ = startTimer(20);
}


void RescaleWidget::
        mouseReleaseEvent ( QMouseEvent * event )
{
    leaveEvent(event);
}


void RescaleWidget::
        paintEvent(QPaintEvent *event)
{
    QPainter painter (this);
    painter.setRenderHints (QPainter::Antialiasing | QPainter::HighQualityAntialiasing);
    //painter.fillPath (path_, QBrush(QColor(125,125,125,125)));
    painter.fillPath (path_, QColor(220,220,220,200));
    painter.strokePath (path_, QPen(
                            QColor(100,100,100,200),
                            hasFocus () ? 1.6 : .8,
                            Qt::SolidLine,
                            Qt::RoundCap,
                            Qt::RoundJoin));

    //painter.drawImage(QPoint(),qimage_);

    QWidget::paintEvent (event);
}


void RescaleWidget::
        resizeEvent ( QResizeEvent * )
{
    QPolygon poly = recreatePolygon ();
    setMask(growRegion(poly));
}


void RescaleWidget::
        painting ()
{
    // draw nothing on the RenderView canvas

    // draw 2D widget region with OpenGL
    HudGlWidget::painting();
}


void RescaleWidget::
        paintWidgetGl2D ()
{
    if ( 0 == "draw 2D widget region with OpenGL")
    {
        glColor4f (1,0,0,0.5);
        glBegin (GL_TRIANGLE_STRIP);
        glVertex2f (0,0);
        glVertex2f (1,0);
        glVertex2f (0,1);
        glVertex2f (1,1);
        glEnd ();

        image_.directDraw();
    }
}


QPolygon RescaleWidget::
        recreatePolygon ()
{
    QPoint o = rect().center();
    QPoint x = scalex_ * (rect().topRight() - rect().topLeft())/4;
    QPoint y = scaley_ * (rect().bottomLeft() - rect().topLeft())/4;

    QPolygon poly;
    float d = 0.3;
    float r = 0.1;
    poly.push_back( o - r*x - r*y);
    poly.push_back( o - d*x - y);
    poly.push_back( o + d*x - y);
    poly.push_back( o + r*x - r*y);
    poly.push_back( o - d*y + x);
    poly.push_back( o + d*y + x);
    poly.push_back( o + r*x + r*y);
    poly.push_back( o + d*x + y);
    poly.push_back( o - d*x + y);
    poly.push_back( o - r*x + r*y);
    poly.push_back( o + d*y - x);
    poly.push_back( o - d*y - x);

    if ( 0 == "push a circle" )
    {
        float N = max(width(), height());
        for (float i=0; i<N; ++i)
        {
            float a = 2*M_PI*(i/N);
            poly.push_back(o + x*cos(a) + y*sin(a));
        }
    }

    path_ = QPainterPath();
    path_.addPolygon(poly);

    update();

    return poly;
}


void RescaleWidget::
        updateModel()
{
    bool success1, success2;

    Heightmap::Position last = view_->getPlanePos( mapToParent(dragSource_), &success1);
    Heightmap::Position current = view_->getPlanePos( mapToParent(lastPos_), &success2);

    QPointF d = lastPos_ - dragSource_;
    float dx = d.x() / width();
    float dy = -d.y() / height();
    dx = max(-.1f, min(.1f, dx));
    dy = max(-.1f, min(.1f, dy));
    scalex_ *= exp2f(dx);
    scaley_ *= exp2f(dy);
    scalex_ = max(0.5f, min(1.8f, scalex_));
    scaley_ = max(0.5f, min(1.8f, scaley_));

    if (success1 && success2)
    {
        float r = DIRECT_RESCALING ? 4 : .1;
        float dt = r*(current.time - last.time)*view_->model->xscale/view_->model->_pz;
        float ds = r*(current.scale - last.scale)*view_->model->zscale/view_->model->_pz;

        Tools::Commands::pCommand cmd( new Tools::Commands::ZoomCameraCommand(view_->model, dt, ds, 0.f ));
        view_->model->project()->commandInvoker()->invokeCommand( cmd );
    }

    //dragSource_ = event->pos();
    recreatePolygon();
}


} // namespace Widgets
} // namespace Tools
