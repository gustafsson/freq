#ifndef TOOLS_WIDGETS_PANWIDGET_H
#define TOOLS_WIDGETS_PANWIDGET_H

#include <QWidget>

namespace Tools {
class RenderView;
namespace Widgets {

class PanWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PanWidget(RenderView* view);
    
protected:
    void leaveEvent ( QEvent * event );
    void mouseMoveEvent ( QMouseEvent * event );
    void mousePressEvent ( QMouseEvent * event );
    void mouseReleaseEvent ( QMouseEvent * event );
    void paintEvent ( QPaintEvent *event );
    void resizeEvent ( QResizeEvent * event );

private:
    void recreatePolygon ();

    RenderView* view_;
    QPainterPath path_;
    QPoint dragSource_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_PANWIDGET_H
