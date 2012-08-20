#include "overlaywidget.h"

#include <QMouseEvent>
#include <QGraphicsProxyWidget>

#include "tools/renderview.h"

namespace Tools {
namespace Widgets {


OverlayWidget::OverlayWidget(RenderView *scene)
    :   scene_(scene)
{
    // Transparent Widget background (alpha-channel is 0)
    this->setPalette(QPalette(QPalette::Window, QColor(255,0,0,0)));

    proxy_ = new QGraphicsProxyWidget(0, Qt::Window);
    proxy_->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy_->setWidget( this );
    proxy_->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy_->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    proxy_->setZValue( 1e10 );
    scene->addItem( proxy_ );

    sceneSection_ = scene->toolSelector()->parentTool();
    sceneSection_->installEventFilter( this );
    setMouseTracking( true ); // setCursor with mask doesn't work with QGraphicsProxyWidget

    connect(scene, SIGNAL(painting()), SLOT(paintGl()));
}


QRect OverlayWidget::
        sceneRect()
{
    return QRect(sceneSection_->pos(), sceneSection_->size());
}


bool OverlayWidget::
        eventFilter(QObject *o, QEvent *e)
{
    if (o == sceneSection_ && e->type()==QEvent::Resize)
        updatePosition();

    return false;
}


} // namespace Widgets
} // namespace Tools
