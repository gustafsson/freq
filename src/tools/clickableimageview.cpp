#include "clickableimageview.h"

#include <QPainter>
#include <QImage>
#include <QBitmap>
#include <QDesktopServices>
#include <QUrl>
#include <QMouseEvent>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QBoxLayout>
#include "renderview.h"
#include <QGLWidget>

namespace Tools {

// imagefile = ":/icons/image.png"
// url = "http://muchdifferent.com/?page=signals"

ClickableImageView::
        ClickableImageView(RenderView *parent, QString imagefile, QString url)
            :
    QWidget(),
    image(imagefile),
    url(url)
{
    setFixedSize( image.size() );

//    parentwidget = parent->toolSelector()->parentTool();
//    this->setParent(parentwidget);

    // Transparent Widget background (alpha-channel is 0)
    this->setPalette(QPalette(QPalette::Window, QColor(255,0,0,0)));

    proxy = new QGraphicsProxyWidget(0, Qt::Window);
    proxy->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy->setWidget( this );
    proxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    proxy->setZValue( 1e10 );
    parent->addItem( proxy );

    //parentwidget = parent->glwidget;
    parentwidget = parent->toolSelector()->parentTool();
    parentwidget->installEventFilter( this );
    setMouseTracking( true ); // setCursor with mask doesn't work with QGraphicsProxyWidget

    connect(parent, SIGNAL(painting()), SLOT(paintGl()));
}


bool ClickableImageView::
        eventFilter(QObject *o, QEvent *e)
{
    if (o == parentwidget && e->type()==QEvent::Resize)
    {
        QSize s = parentwidget->size() - size();
        QPoint p = parentwidget->pos();
        move(p.x() + s.width(), p.y() + s.height());
        image.move(QPointF(
                       parentwidget->width() - width(),
                       0.f));
    }

    return false;
}


void ClickableImageView::
        mousePressEvent(QMouseEvent* e)
{
    if (mask().contains(e->pos()))
        QDesktopServices::openUrl(QUrl(url));
    else
        e->ignore();
}


void ClickableImageView::
        mouseMoveEvent(QMouseEvent * e)
{
    Qt::CursorShape shape = mask().contains(e->pos()) ? Qt::PointingHandCursor : Qt::ArrowCursor;

    if (proxy->cursor().shape() != shape)
        proxy->setCursor( shape );
}


void ClickableImageView::
        paintEvent(QPaintEvent*)
{
    //QPainter painter(this);
    //painter.drawImage( 0, 0, *image);
}


void ClickableImageView::
paintGl()
{
    //image.drawImage(parentWidget()->width(), parentWidget()->height());
}


} // namespace Tools
