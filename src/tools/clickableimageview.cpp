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
    url(url),
    image(new QImage(imagefile))
{
    setFixedSize( image->size() );
    setMask( QRegion(0, 0, image->width(), image->height()) - QBitmap::fromImage( image->alphaChannel() ) );

    proxy = new QGraphicsProxyWidget(0, Qt::Window);
    proxy->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy->setWidget( this );
    proxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    proxy->setZValue( 1e10 );
    proxy->setVisible(true);
    parent->addItem( proxy );

    parentwidget = parent->glwidget;
    parentwidget->installEventFilter( this );
    setMouseTracking( true ); // setCursor with mask doesn't work with QGraphicsProxyWidget
}


bool ClickableImageView::
        eventFilter(QObject *o, QEvent *e)
{
    if (o == parentwidget && e->type()==QEvent::Resize)
    {
        QSize s = parentwidget->size() - size();
        move(s.width(), s.height());
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
    QPainter painter(this);
    painter.drawImage( 0, 0, *image);
}


} // namespace Tools
