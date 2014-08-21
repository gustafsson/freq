#include "overlaywidget.h"

#include <QMouseEvent>
#include <QGraphicsProxyWidget>
#include <QTextStream>
#include <QMetaEnum>

#include "tools/renderview.h"

namespace Tools {
namespace Widgets {


OverlayWidget::OverlayWidget(QGraphicsScene *scene, QWidget* sceneSection)
    :   scene_(scene)
{
    // Qt::WA_NoBackground messes up caches, mimic Qt::WA_NoBackground
//    setPalette(QPalette(QPalette::Window, QColor(0,0,0,0)));
//    setAttribute(Qt::WA_NoBackground);
    setAttribute(Qt::WA_TranslucentBackground);

//    proxy_ = new QGraphicsProxyWidget(0, Qt::Window);
    proxy_ = new QGraphicsProxyWidget;
//    proxy_->setFlag(QGraphicsItem::ItemSendsGeometryChanges);
    proxy_->setWidget( this );
//    proxy_->setWindowFlags( Qt::FramelessWindowHint );
    proxy_->setZValue( -10 );

    // QGraphicsItem::NoCache renders text at HiDPI but doesn't work for other shapes
    proxy_->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
//    proxy_->setCacheMode(QGraphicsItem::ItemCoordinateCache);
//    proxy_->setCacheMode(QGraphicsItem::NoCache);

    scene->addItem( proxy_ );

    sceneSection_ = sceneSection;
    sceneSection_->installEventFilter( this );
    setMouseTracking( true ); // setCursor with mask doesn't work with QGraphicsProxyWidget
}


QRect OverlayWidget::
        sceneRect()
{
    return QRect(sceneSection_->pos(), sceneSection_->size());
}


void OverlayWidget::
        updatePosition ()
{
    QRect r = sceneRect();
    move(r.topLeft());
    resize(r.size());
}


QString eventToString(const QEvent * ev)
{
   static int eventEnumIndex = QEvent::staticMetaObject
         .indexOfEnumerator("Type");
   QString r;
   QTextStream str;
   str.setString (&r);
   str << "QEvent";
   if (ev) {
      QString name = QEvent::staticMetaObject
            .enumerator(eventEnumIndex).valueToKey(ev->type());
      if (!name.isEmpty()) str << name; else str << ev->type();
   } else {
      str << (void*)ev;
   }
   str.flush ();
   return r;
}


bool OverlayWidget::
        event(QEvent *e)
{
    if (e->type () == QEvent::UpdateRequest)
        scene_->update ();

//    TaskInfo("event %s", eventToString(e).toStdString ().c_str ());
    return QWidget::event (e);
}


bool OverlayWidget::
        eventFilter(QObject *o, QEvent *e)
{
//    TaskInfo("eventFilter %s", eventToString(e).toStdString ().c_str ());

    if (o == sceneSection_ && e->type()==QEvent::Resize)
    {
        // Need to recreate the cache when resizing if using QGraphicsItem::ItemCoordinateCache
        if (proxy_->cacheMode () == QGraphicsItem::ItemCoordinateCache)
            proxy_->setCacheMode(QGraphicsItem::ItemCoordinateCache);

        updatePosition();
    }

    return QWidget::eventFilter(o,e);
}


} // namespace Widgets
} // namespace Tools
