#include "graphicsview.h"
#include "renderview.h"

#include <TaskTimer.h>
#include <demangle.h>

#include <QEvent>
#include <QTimerEvent>
#include <QMouseEvent>
#include <QHBoxLayout>
#include <QGraphicsProxyWidget>

//#define DEBUG_EVENTS
#define DEBUG_EVENTS if(0)

namespace Tools
{

GraphicsView::
        GraphicsView(QGraphicsScene* scene)
    :   QGraphicsView(scene)
{
    setWindowTitle(tr("Boxes"));
    //setRenderHints(QPainter::SmoothPixmapTransform);
    //setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform | QPainter::TextAntialiasing);

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);


    setRenderHints(renderHints() | QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    //view.show();
    //view.setWindowTitle("Embedded Dialogs Demo");

    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    //view->tool_selector.reset( new Support::ToolSelector(view->glwidget));
    QGraphicsProxyWidget* toolProxy = new QGraphicsProxyWidget();
    toolParent = new QWidget();

    // Make all child widgets occupy the entire area
    toolParent->setLayout(new QHBoxLayout());
    toolParent->layout()->setMargin(0);

    toolProxy->setWidget( toolParent );
    toolProxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    toolProxy->setZValue(-1e30);
    toolParent->setWindowOpacity(0);
    scene->addItem( toolProxy );
}


GraphicsView::
        ~GraphicsView()
{
    if (scene())
        delete scene();
}


bool GraphicsView::
        event ( QEvent * e )
{
    DEBUG_EVENTS TaskTimer tt("GraphicsView event %s %d", vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsView::event(e);
    DEBUG_EVENTS TaskTimer("GraphicsView event %s info %d %d", vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}


bool GraphicsView::
        eventFilter(QObject* o, QEvent* e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsView eventFilter %s %s %d", vartype(*o).c_str(), vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsView::eventFilter(o, e);
    DEBUG_EVENTS TaskTimer("GraphicsView eventFilter %s %s info %d %d", vartype(*o).c_str(), vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}


void GraphicsView::timerEvent(QTimerEvent *e){
    DEBUG_EVENTS TaskTimer tt("GraphicsView timerEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::timerEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView timerEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::childEvent(QChildEvent *e){
    DEBUG_EVENTS TaskTimer tt("GraphicsView childEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::childEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView childEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::customEvent(QEvent *e){
    DEBUG_EVENTS TaskTimer tt("GraphicsView customEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::customEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView customEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::mousePressEvent( QMouseEvent* e )
{
    DEBUG_EVENTS TaskTimer tt("GraphicsView mousePressEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::mousePressEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView mousePressEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::mouseMoveEvent(QMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsView mouseMoveEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::mouseMoveEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView mouseMoveEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::mouseReleaseEvent(QMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsView mouseReleaseEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsView::mouseReleaseEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsView mouseReleaseEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsView::drawBackground(QPainter *painter, const QRectF &rect)
{
    QGraphicsView::drawBackground( painter, rect );

    //        view->makeCurrent();
    //setTransform( QTransform().translate(1000,1) );
    //setTransform( QTransform() );
    //setTransform( QTransform().translate(view->_qx*5e18,1).scale(0.5,0.5  ).rotate(45));
    //setTransform( QTransform().translate(0,1e15*view->_qx).scale(0.5,0.5  ).rotate(45));
    //setTransform( QTransform().translate(0,0).scale(0.5,0.5  ).rotate(45));

    /*ViewportAnchor a = transformationAnchor();
    Qt::Alignment al = alignment();
    setAlignment(Qt::AlignLeft | Qt::AlignTop);*/

    //rotate(0.1);
    //setTransform( QTransform().rotate(view->_qx, Qt::YAxis) );
    //setTransform( QTransform().rotate(view->_qx, Qt::YAxis) );
    //setTransform( QTransform().rotate(0.123, Qt::YAxis), true );
    //setTransform( QTransform().rotate(0.157, Qt::ZAxis), true );
    //translate(view->_qx, 0);
    //scale();
    //setTransform( QTransform().translate(view->_qx*1e18,1).scale(0.5,0.5  ));
//        setTransform( QTransform().translate(view->_qx*5e18,1).scale(0.5,0.5  ));
    //setTransform( QTransform().translate(1,1) );
}


void GraphicsView::resizeEvent(QResizeEvent *event) {
    //float h = event->size().height();
    if (scene())
        scene()->setSceneRect(QRectF(0, 0, event->size().width(), event->size().height()));

    toolParent->resize( event->size() );
}


} // namespace Tools
