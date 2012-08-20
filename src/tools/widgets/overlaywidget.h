#ifndef TOOLS_WIDGETS_GRAPHWIDGET_H
#define TOOLS_WIDGETS_GRAPHWIDGET_H

#include <QWidget>
#include "tools/support/drawimage.h"

class QGraphicsScene;

namespace Tools {
class RenderView;

namespace Widgets {

class OverlayWidget : public QWidget
{
    Q_OBJECT
public:
    explicit OverlayWidget(RenderView *parent);

    QRect sceneRect();

signals:
    
public slots:

protected:
    virtual void updatePosition() = 0;

private:
    bool eventFilter(QObject *o, QEvent *e);

    QWidget* sceneSection_;
    QGraphicsScene* scene_;
    QGraphicsProxyWidget* proxy_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_GRAPHWIDGET_H
