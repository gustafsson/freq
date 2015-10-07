#ifndef TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H
#define TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H

#include "overlaywidget.h"
#include "tools/commands/commandinvoker.h"

#include <QList>
#include <QMouseEvent>

namespace Tools {
namespace Widgets {

class WidgetOverlayController: public OverlayWidget
{
public:
    WidgetOverlayController(
            QGraphicsScene* scene,
            RenderView* view,
            Tools::Commands::CommandInvoker* commandInvoker,
            Tools::Support::ToolSelector* tool_selector);

    ~WidgetOverlayController();

    void enterEvent ( QEvent * ) override;
    void leaveEvent ( QEvent * ) override;
    void keyPressEvent ( QKeyEvent * ) override;
    void keyReleaseEvent ( QKeyEvent * ) override;
    void mouseMoveEvent ( QMouseEvent * event ) override;
    void mousePressEvent ( QMouseEvent * event ) override;
    void mouseReleaseEvent ( QMouseEvent * event ) override;

protected:
    void updatePosition() override;

private:
    void setupLayout();
    void setupLayoutCenter();
    void setupLayoutRightAndBottom();

    bool updateFocusWidget(QKeyEvent *e);
    void sendMouseProxyEvent( QMouseEvent * event );

    QWidget *pan_, *rescale_, *rotate_, *proxy_mousepress_;
    RenderView* view_;
    Tools::Commands::CommandInvoker* commandInvoker_;
    QMouseEvent child_event_;
    QPoint lastMousePos_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H
