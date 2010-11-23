#include "commentview.h"
#include "ui_commentview.h"

#include "renderview.h"
#include "ui/mousecontrol.h"

#include <QGraphicsProxyWidget>
#include <QMoveEvent>
#include <demangle.h>
#include <QWheelEvent>
#include <QPaintEvent>

namespace Tools {

CommentView::CommentView(QWidget *parent) :
        QWidget(parent),
    ui(new Ui::CommentView),
    keep_pos(false),
    scroll_scale(1),
    z_hidden(false)
{
    //
    ui->setupUi(this);

    QAction *closeAction = new QAction(tr("C&lose"), this);
    closeAction->setShortcut(tr("Ctrl+D"));
    connect(closeAction, SIGNAL(triggered()), SLOT(close()));
    addAction(closeAction);
    setContextMenuPolicy(Qt::ActionsContextMenu);
}


CommentView::~CommentView()
{
    delete ui;
}


/*bool CommentView::event ( QEvent * e )
{
    TaskTimer tt("CommentView event %s %d", vartype(*e).c_str(), e->isAccepted());
    bool r = QWidget::event(e);
    tt.info("CommentView event %s info %d %d", vartype(*e).c_str(), r, e->isAccepted());
    return r;
}*/


void CommentView::
        mousePressEvent(QMouseEvent *event)
{
    QPoint gp = proxy->sceneTransform().map(event->globalPos());
    if (event->buttons() & Qt::LeftButton)
    {
        if (event->modifiers() == 0)
        {
            dragPosition = gp;
            event->accept();
        }
        else if (event->modifiers().testFlag(Qt::ControlModifier))
        {
            resizePosition = -QPoint(width(), height()) + QPoint(gp.x(), -gp.y());
            event->accept();
        }
    }
    update();
}


void CommentView::
        mouseMoveEvent(QMouseEvent *event)
{
    QPoint gp = proxy->sceneTransform().map(event->globalPos());
    if (event->buttons() & Qt::LeftButton)
    {
        if (event->modifiers() == 0)
        {
            move(gp - dragPosition);
            dragPosition = gp;
            event->accept();
        }
        else if (event->modifiers().testFlag(Qt::ControlModifier))
        {
            QPoint sz = QPoint(gp.x(),-gp.y()) - resizePosition;
            resize(sz.x(), sz.y());
            event->accept();
        }
    }
    update();
}


void CommentView::
        wheelEvent(QWheelEvent *e)
{
    if (e->delta()>0)
        scroll_scale *= 1.1;
    else
        scroll_scale /= 1.1;
    update();
}


void CommentView::
        resizeEvent(QResizeEvent *)
{
    keep_pos = true;

    QRect r = ui->textEdit->geometry();
    r.setTop(r.top()-1);
    r.setLeft(r.left()-1);
    r.setRight(r.right()+2);
    r.setBottom(r.bottom()+1);
    QPoint x = r.bottomRight() - r.bottomLeft();
    x.setX(x.x()-1);
    QPoint b = r.bottomLeft();
    b.setY(b.y()+1);
    QPoint y = QPoint(0, frameGeometry().bottom()-b.y());
    QPoint h = r.topLeft() - r.bottomLeft();
    QPointF h0(0, 1);
    QPointF x0(1, 0);
    poly.clear();
    poly.push_back(b - 2*h0);
    poly.push_back(b + 2*x0);
    poly.push_back(b + 0.1f*y + 0.2f*x);
    ref_point = b + y + 0.1f*x;
    poly.push_back(ref_point);
    poly.push_back(b + 0.1f*y + 0.4f*x);
    poly.push_back(b + x - x0);
    poly.push_back(b + x - h0);
    poly.push_back(b + x + h + 0.5f*h0);
    poly.push_back(b + x + h - 0.5f*x0);
    poly.push_back(b + h + 2*x0);
    poly.push_back(b + h + h0);

    /*r.setTop(r.top()-1);
    r.setLeft(r.left()-1);
    r.setRight(r.right()+1);
    r.setBottom(r.bottom());*/
    r.setLeft(r.left()+1);
    r.setRight(r.right()-2);

    QRegion maskedRegion = r;
    maskedRegion |= QRegion( poly.toPolygon() );
    maskedRegion -= QRegion(0,0,1,1);
    maskedRegion -= QRegion(r.right()-2,0,2,1);
    maskedRegion -= QRegion(0,r.bottom()-2,1,1);
    maskedRegion -= QRegion(r.right()-2,r.bottom()-2,1,1);
    maskedRegion |= maskedRegion.translated(-1, 0);
    maskedRegion |= maskedRegion.translated(1, 0);
    maskedRegion |= maskedRegion.translated(0, 1);
    maskedRegion |= maskedRegion.translated(1, 1);
    maskedRegion |= maskedRegion.translated(1, 1);
    setMask(maskedRegion);

    update();
}


void CommentView::
        paintEvent(QPaintEvent *e)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setBrush(QApplication::palette().color(
            hasFocus() ? QPalette::Active : QPalette::Inactive,
            QPalette::Base ));
    painter.drawPolygon(poly);

    QWidget::paintEvent(e);
}


QSize CommentView::
        sizeHint() const
{
    return QSize(200,200);
}


void CommentView::
        updatePosition()
{
    // moveEvent can't be used when updating the reference position while moving
    if (!proxy->pos().isNull())
    {
        if (!keep_pos)
        {
            QPointF c = proxy->sceneTransform().map(QPointF(ref_point));

            float h = proxy->scene()->height();
            c.setY( h - 1 - c.y() );

            pos = view->getHeightmapPos( c );

            /*            float x, y;
        Ui::MouseControl::planePos( c.x(), c.y(), x, y, view->xscale );
        pos.time = x;
        pos.scale = y;*/
        }

        keep_pos = false;

        move(0,0);
        proxy->scene()->update();
        update();
    }

    double z;
    QPointF pt = view->getScreenPos( Heightmap::Position( pos.time, pos.scale), &z );

    proxy->setZValue(-z);

    if (z>0)
    {
        z *= 0.5;

        if (-1 > view->_pz)
            z += -log(-view->_pz);

        if (z < 1)
            z = 1;
    }

    if (z<=0)
    {
        if (isVisible())
        {
            z_hidden = true;
            hide();
        }
    }
    else if (z_hidden)
    {
        show();
        z_hidden = false;
    }


    float rescale = 1.f/sqrt(z);

    rescale*= scroll_scale;

    proxy->setTransform(QTransform()
        .translate(pt.x(), pt.y())
        .scale(rescale, rescale)
        .translate( -ref_point.x(), -ref_point.y() )
        );
}

} // namespace Tools
