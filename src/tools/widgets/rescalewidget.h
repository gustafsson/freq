#ifndef TOOLS_WIDGETS_RESCALEWIDGET_H
#define TOOLS_WIDGETS_RESCALEWIDGET_H

#include "hudglwidget.h"
#include "tools/support/drawimage.h"

namespace Tools {
class RenderView;
namespace Widgets {

class RescaleWidget : public HudGlWidget
{
public:
    explicit RescaleWidget (RenderView*);

protected:
    void timerEvent ( QTimerEvent * e );

    void leaveEvent ( QEvent * event );
    void mouseMoveEvent ( QMouseEvent * event );
    void mousePressEvent ( QMouseEvent * event );
    void mouseReleaseEvent ( QMouseEvent * event );
    void paintEvent (QPaintEvent *e);
    void resizeEvent ( QResizeEvent * event );

    void painting ();
    void paintWidgetGl2D ();

    void recreatePolygon ();
    void updateModel ();

private:
    RenderView* view_;
    QPainterPath path_;
    float scalex_;
    float scaley_;
    QPoint dragSource_;
    QPoint lastPos_;
    Support::DrawImage image_;
    QImage qimage_;
    int updateTimer_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_RESCALEWIDGET_H
