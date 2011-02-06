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
    explicit CommentView(CommentModel* model, QWidget *parent = 0);
    ~CommentView();

    std::string html();
    void setHtml(std::string);

    RenderView* view;
    QGraphicsProxyWidget* proxy;
    CommentModel* model;

    virtual void wheelEvent(QWheelEvent *);
    virtual void resizeEvent(QResizeEvent *);
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void focusInEvent(QFocusEvent *event);
    virtual void focusOutEvent(QFocusEvent *event);
    virtual QSize sizeHint() const;

    bool isThumbnail();

signals:
    void setCommentControllerEnabled( bool );
    void thumbnailChanged( bool );
    void gotFocus();

public slots:
    void updatePosition();
    void updateText();
    void recreatePolygon();
    void thumbnail(bool);

private:
    Ui::CommentView *ui;

    QPoint ref_point;
    QPolygonF poly;
    bool keep_pos;
    bool z_hidden;
    QPoint dragPosition;
    QPoint resizePosition;

    bool testFocus();
};

} // namespace Tools

#endif // COMMENTVIEW_H
