#ifndef CLICKABLEIMAGEVIEW_H
#define CLICKABLEIMAGEVIEW_H

#include <QWidget>
#include <QScopedPointer>

class QImage;
class QGraphicsProxyWidget;

namespace Tools {

class RenderView;

class ClickableImageView : public QWidget
{
    Q_OBJECT
public:
    explicit ClickableImageView(RenderView *parent, QString image, QString url);

    virtual void mousePressEvent(QMouseEvent*);
    virtual void paintEvent(QPaintEvent *e);
    virtual bool eventFilter(QObject *o, QEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *);

private:
    QString url;
    QWidget *parentwidget;
    QGraphicsProxyWidget* proxy;

    QScopedPointer<QImage> image;
};

} // namespace Tools

#endif // GETCUDAVIEW_H
