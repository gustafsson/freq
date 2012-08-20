#ifndef TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H
#define TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H

#include "overlaywidget.h"

#include <QList>
#include <QPointer>

namespace Tools {
namespace Widgets {

class WidgetOverlayController: public OverlayWidget
{
public:
    WidgetOverlayController(RenderView* view);
    ~WidgetOverlayController();

protected:
    void updatePosition();

private:
    void setupLayout();
    void setupLayoutCenter();
    void setupLayoutRightAndBottom();

    RenderView* view_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_WIDGETOVERLAYCONTROLLER_H
