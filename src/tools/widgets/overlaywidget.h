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
    // QWidget* sceneSection = tool_selector->parentTool()
    explicit OverlayWidget(QGraphicsScene *parent, QWidget* sceneSection);

    QRect sceneRect();

signals:
    
public slots:

protected:
    virtual void updatePosition();

private:
    bool event(QEvent *e) override;
    bool eventFilter(QObject *o, QEvent *e) override;

    QWidget* sceneSection_;
    QGraphicsScene* scene_;
    QGraphicsProxyWidget* proxy_;
};

} // namespace Widgets
} // namespace Tools

#endif // TOOLS_WIDGETS_GRAPHWIDGET_H
