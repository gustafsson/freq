#ifndef TOOLS_WIDGETS_ROTATEWIDGET_H
#define TOOLS_WIDGETS_ROTATEWIDGET_H

#include <QWidget>
#include "tools/commands/commandinvoker.h"

namespace Tools {
class RenderView;
namespace Widgets {

class RotateWidget : public QWidget
{
    Q_OBJECT
public:
    explicit RotateWidget(RenderView *view, Tools::Commands::CommandInvoker* commandInvoker);
    
protected:
    void leaveEvent ( QEvent * event );
    void mouseMoveEvent ( QMouseEvent * event );
    void mousePressEvent ( QMouseEvent * event );
    void mouseReleaseEvent ( QMouseEvent * event );
    void paintEvent ( QPaintEvent *event );
    void resizeEvent ( QResizeEvent * event );

private:
    void recreatePolygon ();
    QPolygon circleShape ();
    QPolygon bunkShape ();

    RenderView* view_;
    Tools::Commands::CommandInvoker* commandInvoker_;
    QPainterPath path_;
    QPoint dragSource_;
    bool mouseMoved_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_ROTATEWIDGET_H
