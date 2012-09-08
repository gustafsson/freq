#include "overlaywidget.h"

#include <QMouseEvent>
#include <QGraphicsProxyWidget>

#include "tools/renderview.h"

namespace Tools {
namespace Widgets {


OverlayWidget::OverlayWidget(RenderView *scene)
    :   scene_(scene)
{
    setAttribute(Qt::WA_NoBackground);

    proxy_ = new QGraphicsProxyWidget(0, Qt::Window);
    proxy_->setFlag(QGraphicsItem::ItemSendsGeometryChanges);
    proxy_->setWidget( this );
    proxy_->setWindowFlags( Qt::FramelessWindowHint );
    proxy_->setZValue( 1e10 );
    scene->addItem( proxy_ );

    sceneSection_ = scene->toolSelector()->parentTool();
    sceneSection_->installEventFilter( this );
    setMouseTracking( true ); // setCursor with mask doesn't work with QGraphicsProxyWidget
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
    {
        updatePosition();
        // recreate the cache
        //proxy_->setCacheMode(QGraphicsItem::ItemCoordinateCache);
        proxy_->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
    }

    return false;
}


} // namespace Widgets
} // namespace Tools
