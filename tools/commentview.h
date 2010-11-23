#ifndef COMMENTVIEW_H
#define COMMENTVIEW_H

#include "commentmodel.h"

#include <QWidget>

namespace Ui {
    class CommentView;
}

namespace Tools {

class RenderView;

class CommentView : public QWidget
{
    Q_OBJECT

public:
    explicit CommentView(QWidget *parent = 0);
    ~CommentView();

    //QString text();
    Heightmap::Position pos;

    RenderView* view;
    QGraphicsProxyWidget* proxy;

    virtual void wheelEvent(QWheelEvent *);
    virtual void resizeEvent(QResizeEvent *);
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual QSize sizeHint() const;

signals:
    void setCommentControllerEnabled( bool );

public slots:
    void updatePosition();

private:
    Ui::CommentView *ui;

    QPoint ref_point;
    QPolygonF poly;
    bool keep_pos;
    float scroll_scale;
    bool z_hidden;
    QPoint dragPosition;
    QPoint resizePosition;
};

} // namespace Tools

#endif // COMMENTVIEW_H
