#include "panwidget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QApplication>

#include "hudglwidget.h"
#include "sawe/project.h"
#include "tools/support/renderviewinfo.h"
#include "tools/commands/movecameracommand.h"

namespace Tools {
namespace Widgets {


PanWidget::PanWidget(RenderView *view) :
    view_(view)
{
    setMinimumSize(70,70);
    setCursor(Qt::OpenHandCursor);
#ifdef __APPLE__
    setToolTip("Click and drag to pan [hold ctrl]");
#else
    setToolTip("Click and drag to pan [hold shift]");
#endif
}


void PanWidget::
        leaveEvent ( QEvent * )
{
    // Restore closed hand cursor, called from mouseReleaseEvent
    QApplication::restoreOverrideCursor();
}


void PanWidget::
        mouseMoveEvent ( QMouseEvent * event )
{
    bool success1, success2;

    Tools::Support::RenderViewInfo r(view_);
    Heightmap::Position last = r.getPlanePos( mapToParent(dragSource_), &success1);
    Heightmap::Position current = r.getPlanePos( mapToParent(event->pos()), &success2);

    if (success1 && success2)
    {
        float dt = current.time - last.time;
        float ds = current.scale - last.scale;

        Tools::Commands::pCommand cmd( new Tools::Commands::MoveCameraCommand(view_->model, -dt, -ds ));
        view_->model->project()->commandInvoker()->invokeCommand( cmd );
    }

    dragSource_ = event->pos();
}


void PanWidget::
        mousePressEvent ( QMouseEvent * event )
{
    QApplication::setOverrideCursor(Qt::ClosedHandCursor);
    dragSource_ = event->pos();
}


void PanWidget::
        mouseReleaseEvent ( QMouseEvent * event )
{
    leaveEvent(event);
}


void PanWidget::
        paintEvent ( QPaintEvent * event )
{
    QPainter painter (this);
    painter.setRenderHints (QPainter::Antialiasing | QPainter::HighQualityAntialiasing);
    painter.fillPath (path_, QColor(220,220,220,200));
    painter.strokePath (path_, QPen(
                            QColor(100,100,100,200),
                            hasFocus () ? 1.6 : .8,
                            Qt::SolidLine,
                            Qt::RoundCap,
                            Qt::RoundJoin));

    QWidget::paintEvent(event);
}



void PanWidget::
        resizeEvent ( QResizeEvent *)
{
    recreatePolygon();
}


void PanWidget::
        recreatePolygon ()
{
    QPoint o = rect().center();
    QPoint x = (rect().topRight() - rect().topLeft())/4;
    QPoint y = (rect().bottomLeft() - rect().topLeft())/4;

    QPolygon poly;
    float R = 1.5;
    float r = 0.25;
    poly.push_back( o - R*y );
    poly.push_back( o + r*x - r*y);
    poly.push_back( o + R*x );
    poly.push_back( o + r*x + r*y);
    poly.push_back( o + R*y );
    poly.push_back( o - r*x + r*y);
    poly.push_back( o - R*x );
    poly.push_back( o - r*x - r*y);

    path_ = QPainterPath();
    path_.addPolygon(poly);

    setMask(HudGlWidget::growRegion(poly));
    update();
}


} // namespace Widgets
} // namespace Tools
