#ifndef CLICKABLEIMAGEVIEW_H
#define CLICKABLEIMAGEVIEW_H

#include <QWidget>

#include "support/drawimage.h"

class QImage;
class QGraphicsProxyWidget;

namespace Tools {

class RenderView;

class ClickableImageView : public QWidget
{
    Q_OBJECT
public:
    explicit ClickableImageView(RenderView *parent, QString image=":/icons/muchdifferent.png", QString url="http://muchdifferent.com/?page=signals");

    virtual void mousePressEvent(QMouseEvent*);
    virtual void paintEvent(QPaintEvent *e);
    virtual bool eventFilter(QObject *o, QEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *);

private slots:
    void paintGl();

private:
    QWidget *parentwidget;
    QGraphicsProxyWidget* proxy;

    Support::DrawImage image;
    QString url;
};

} // namespace Tools

#endif // GETCUDAVIEW_H
