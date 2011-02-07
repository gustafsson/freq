#ifndef GRAPHICSVIEW_H
#define GRAPHICSVIEW_H

#include <QGraphicsView>
#include <QTimer>
#include <QBoxLayout>

#include "support/toolselector.h"

namespace Tools
{

class GraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit GraphicsView(QGraphicsScene* scene);

    QTimer t;

    ~GraphicsView();

    bool event ( QEvent * e );
    bool eventFilter(QObject* o, QEvent* e);

    void timerEvent(QTimerEvent *e);
    void childEvent(QChildEvent *e);
    void customEvent(QEvent *e);

    void mousePressEvent( QMouseEvent* e );
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void drawBackground(QPainter *painter, const QRectF &rect);

    void resizeEvent(QResizeEvent *event);

    unsigned toolWindows();
    Support::ToolSelector* toolSelector(int index);

    void setLayoutDirection( QBoxLayout::Direction direction );

signals:
    /**
      */
    void layoutChanged( QBoxLayout::Direction direction );

public slots:

private:
    QWidget* layout_widget_;
};

} // namespace Tools

#endif // GRAPHICSVIEW_H
