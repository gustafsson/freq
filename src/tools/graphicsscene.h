#ifndef TOOLS_GRAPHICSSCENE_H
#define TOOLS_GRAPHICSSCENE_H

#include <QGraphicsScene>
#include "timer.h"
#include "renderview.h"

namespace Tools {

class GraphicsScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit GraphicsScene(RenderView* renderview);
    ~GraphicsScene();

    void drawBackground(QPainter *painter, const QRectF &) override;
    void drawForeground(QPainter *painter, const QRectF &) override;
    bool event( QEvent * e ) override;
    bool eventFilter(QObject* o, QEvent* e) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

signals:

public slots:
    void redraw();

private:
    /**
      Adjusting sleep between frames based on fps.
      */
    Timer            last_frame_;
    QPointer<QTimer> update_timer_;
    int              draw_more_ = 0;
    RenderView*      renderview_;

};

} // namespace Tools

#endif // TOOLS_GRAPHICSSCENE_H
