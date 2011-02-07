#include "commentview.h"
#include "ui_commentview.h"

#include "renderview.h"
#include "ui/mousecontrol.h"

// gpumisc
#include <demangle.h>

// qt
#include <QGraphicsProxyWidget>
#include <QMoveEvent>
#include <QWheelEvent>
#include <QPaintEvent>
#include <QPainter>

namespace Tools {

CommentView::CommentView(CommentModel* model, QWidget *parent) :
    QWidget(parent),
    model(model),
    ui(new Ui::CommentView),
    keep_pos(false),
    z_hidden(false)
{
    //
    ui->setupUi(this);

    QAction *closeAction = new QAction(tr("D&elete"), this);
    //closeAction->setShortcut(tr("Ctrl+D"));
    connect(closeAction, SIGNAL(triggered()), SLOT(close()));
    this->setAttribute( Qt::WA_DeleteOnClose );

    QAction *hideAction = new QAction(tr("T&humbnail"), this);
    //hideAction->setShortcut(tr("Ctrl+T"));
    hideAction->setCheckable(true);
    connect(hideAction, SIGNAL(toggled(bool)), SLOT(thumbnail(bool)));
    connect(this, SIGNAL(thumbnailChanged(bool)), hideAction, SLOT(setChecked(bool)));

	connect(ui->textEdit, SIGNAL(textChanged()), SLOT(updateText()));
    addAction(closeAction);
    addAction(hideAction);
    setMouseTracking( true );
	setHtml(model->html);
    //setFocusPolicy(Qt::WheelFocus);
    //ui->textEdit->setFocusProxy(this);
    connect(ui->textEdit, SIGNAL(selectionChanged()), SLOT(recreatePolygon()));
}


CommentView::~CommentView()
{
    delete ui;
}


std::string CommentView::
        html()
{
    return ui->textEdit->toHtml().toLocal8Bit().data();
}


void CommentView::
        setHtml(std::string text)
{
    ui->textEdit->setHtml( QString::fromLocal8Bit( text.c_str() ) );
	model->html = text;
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
    if (model->move_on_hover)
    {
        event->setAccepted( false );
        return;
    }

    if (!mask().contains( event->pos() ))
    {
        event->setAccepted( false );
        return;
    }

    //TaskInfo("CommentView::mousePressEvent");
    if (!testFocus())
    {
        setFocus(Qt::MouseFocusReason);
        recreatePolygon();
        ui->textEdit->setFocus(Qt::MouseFocusReason);
    }


    if (event->buttons() & Qt::LeftButton)
    {
        QPoint gp = proxy->sceneTransform().map(event->globalPos());

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
    view->userinput_update();
}


void CommentView::
        mouseDoubleClickEvent ( QMouseEvent * event )
{
    if (!mask().contains( event->pos() ))
    {
        event->setAccepted( false );
        return;
    }

    thumbnail( !model->thumbnail );
}


void CommentView::
        mouseMoveEvent(QMouseEvent *event)
{    
    bool visible = mask().contains( event->pos() );
    setContextMenuPolicy( visible ? Qt::ActionsContextMenu : Qt::NoContextMenu);

    bool moving = false;
    bool resizing = false;

    if (event->buttons() & Qt::LeftButton)
    {
        moving |= event->modifiers() == 0 && !model->freezed_position;
        resizing |= event->modifiers().testFlag(Qt::ControlModifier);
    }

    moving |= model->move_on_hover;
    if (model->move_on_hover)
        dragPosition = proxy->sceneTransform().map(ref_point);

    if (moving || resizing)
    {
        QPoint gp = proxy->sceneTransform().map(event->globalPos());

        if (moving)
        {
            move(gp - dragPosition);
            dragPosition = gp;
            event->accept();
        }
        else if (resizing)
        {
            QPoint sz = QPoint(gp.x(),-gp.y()) - resizePosition;
            resize(sz.x(), sz.y());
            event->accept();
        }
        resizePosition = -QPoint(width(), height()) + QPoint(gp.x(), -gp.y());
    }

    update();
    view->userinput_update();
}


void CommentView::
        mouseReleaseEvent(QMouseEvent *event)
{
    emit setCommentControllerEnabled( false );
    QWidget::mouseReleaseEvent(event);
}


void CommentView::
        focusInEvent(QFocusEvent * /*e*/)
{
    //TaskInfo("CommentView::focusInEvent, gotFocus = %d, reason = %d", e->gotFocus(), e->reason());

    recreatePolygon();

    emit gotFocus();
}


void CommentView::
        focusOutEvent(QFocusEvent * /*e*/)
{
    //TaskInfo("CommentView::focusOutEvent, lostFocus = %d, reason = %d", e->lostFocus(), e->reason());

    recreatePolygon();
}


void CommentView::
        wheelEvent(QWheelEvent *e)
{
    if (!mask().contains( e->pos() ))
    {
        e->setAccepted( false );
        return;
    }

    if (e->delta()>0)
        model->scroll_scale *= 1.1;
    else
        model->scroll_scale /= 1.1;

    update();
}


void CommentView::
        resizeEvent(QResizeEvent *)
{
    model->window_size = make_uint2( width(), height() );
    keep_pos = true;

    recreatePolygon();
}


bool CommentView::
        testFocus()
{
    return true;
    //return hasFocus() || ui->textEdit->hasFocus();
}


void CommentView::
        recreatePolygon()
{
    bool create_thumbnail = !testFocus() || model->thumbnail;

    ui->textEdit->setVisible( !create_thumbnail );

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
    ref_point = b + y + 0.1f*x;

    if (!create_thumbnail)
    {
        ui->textEdit->setVisible( true );
        poly.clear();
        poly.push_back(b - 2*h0);
        poly.push_back(b + 2*x0);
        poly.push_back(b + 0.1f*y + 0.2f*x);
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
    }
    else
    {
        ui->textEdit->setVisible( false );
        poly.clear();
        r = QRect();
        float s = (ref_point.x()-1)/2;
        poly.push_back(ref_point);
        poly.push_back(ref_point - s*h0 - s*x0);
        poly.push_back(ref_point - s*h0 + s*x0);
    }


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

    if (create_thumbnail)
        maskedRegion |= QRegion(0,0,1,1);

    setMask(maskedRegion);

    update();
    proxy->update();
    view->userinput_update();
}


void CommentView::
        thumbnail(bool v)
{
    if (model->thumbnail != v)
    {
        model->thumbnail = v;
        emit thumbnailChanged( model->thumbnail );
    }

    recreatePolygon();
}


void CommentView::
        updateText()
{
	model->html = html();
}


void CommentView::
        paintEvent(QPaintEvent *e)
{
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setBrush(QApplication::palette().color(
                testFocus() ? QPalette::Active : QPalette::Inactive,
                QPalette::Base ));
        painter.drawPoint(0,0);
        painter.drawPolygon(poly);

        //TaskInfo("CommentView::paintEvent, %d", testFocus());
    }

    QWidget::paintEvent(e);
}


QSize CommentView::
        sizeHint() const
{
    return QSize(200,200);
}


bool CommentView::
        isThumbnail()
{
    return model->thumbnail;
}


void CommentView::
        updatePosition()
{
    if (!isVisible())
        return;

    // moveEvent can't be used when updating the reference position while moving
    if (!proxy->pos().isNull())
    {
        if (!keep_pos)
        {
            QPointF c = proxy->sceneTransform().map(QPointF(ref_point));

            model->pos = view->getHeightmapPos( c );
        }

        keep_pos = false;

        move(0,0);
        proxy->scene()->update();
        update();
        view->userinput_update();
    }

    double z;
    QPointF pt = view->getScreenPos( model->pos, &z );
    //TaskInfo("model->pos( %g, %g ) -> ( %g, %g, %g )",
    //         model->pos.time, model->pos.scale,
    //         pt.x(), pt.y(), z);

    proxy->setZValue(-z);

    if (z>0)
    {
        z *= 0.5;

        if (-1 > view->model->_pz)
            z += -log(-view->model->_pz);

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

    rescale *= model->scroll_scale;
    //TaskInfo("rescale = %g\tz = %g\tmodel->scroll_scale = %g\tlast_ysize = %g",
    //         rescale, z, model->scroll_scale, view->last_ysize );

    proxy->setTransform(QTransform()
        .translate(pt.x(), pt.y())
        .scale(rescale, rescale)
        .translate( -ref_point.x(), -ref_point.y() )
        );
}

} // namespace Tools
