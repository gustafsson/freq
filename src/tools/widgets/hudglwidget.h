#ifndef TOOLS_WIDGETS_HUDWIDGET_H
#define TOOLS_WIDGETS_HUDWIDGET_H

#include <QWidget>

#include "tools/support/drawimage.h"

namespace Tools {
class RenderView;

namespace Widgets {

class HudGlWidget : public QWidget
{
    Q_OBJECT
public:
    explicit HudGlWidget (RenderView *view);
    
    static QRegion growRegion(const QRegion& r, int radius=2);

signals:
    
public slots:

protected slots:
    /**
     * @brief paint2D sets up the OpenGL viewport with an orthonormal (0 to 1) projection for this widget and calls paintWidgetGl2D
     */
    virtual void painting ();

protected:
    /**
     * @brief paintWidgetGl2D is called by painting
     */
    virtual void paintWidgetGl2D () = 0;

private:
    RenderView *view_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_HUDWIDGET_H
