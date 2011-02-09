#include "commentview.h"
#include "ui_commentview.h"

#include "renderview.h"
#include "ui/mousecontrol.h"

#include "sawe/project.h"

// gpumisc
#include <demangle.h>

// qt
#include <QGraphicsProxyWidget>
#include <QMoveEvent>
#include <QWheelEvent>
#include <QPaintEvent>
#include <QPainter>

namespace Tools {

CommentView::CommentView(ToolModelP modelp, QWidget *parent) :
    QWidget(parent),
    modelp(modelp),
    ui(new Ui::CommentView),
    keep_pos(false),
    z_hidden(false),
    lastz(6)
{
    //
    ui->setupUi(this);

    BOOST_ASSERT( dynamic_cast<CommentModel*>(modelp.get() ));

    QAction *closeAction = new QAction(tr("D&elete"), this);
    //closeAction->setShortcut(tr("Ctrl+D"));
    connect(closeAction, SIGNAL(triggered()), SLOT(close()));

    QAction *hideAction = new QAction(tr("T&humbnail"), this);
    //hideAction->setShortcut(tr("Ctrl+T"));
    hideAction->setCheckable(true);
    connect(hideAction, SIGNAL(toggled(bool)), SLOT(thumbnail(bool)));
    connect(this, SIGNAL(thumbnailChanged(bool)), hideAction, SLOT(setChecked(bool)));

	connect(ui->textEdit, SIGNAL(textChanged()), SLOT(updateText()));
    addAction(closeAction);
    addAction(hideAction);
    setMouseTracking( true );
    setHtml(model()->html);
    //ui->textEdit->setFocusProxy(this);
    connect(ui->textEdit, SIGNAL(selectionChanged()), SLOT(recreatePolygon()));
}


CommentView::~CommentView()
{
    TaskInfo("~CommentView");
    delete ui;

    model()->removeFromRepo();
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
    model()->html = text;
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
    if (model()->move_on_hover)
    {
        model()->move_on_hover = false;
        model()->screen_pos.x = -2;
        event->setAccepted( false );
        return;
    }

    if (!mask().contains( event->pos() ))
    {
        setEditFocus( false );
        event->setAccepted( false );
        return;
    }

    //TaskInfo("CommentView::mousePressEvent");
    if (!testFocus())
    {
        setEditFocus(true);
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

    thumbnail( !model()->thumbnail );
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
        moving |= event->modifiers() == 0 && !model()->freezed_position;
        resizing |= event->modifiers().testFlag(Qt::ControlModifier);
    }

    moving |= model()->move_on_hover;
    if (model()->move_on_hover)
        dragPosition = proxy->sceneTransform().map(ref_point);

    if (moving || resizing)
    {
        QPoint gp = proxy->sceneTransform().map(event->globalPos());

        if (moving)
        {
            move(gp - dragPosition);
            QPoint global_ref_pt = proxy->sceneTransform().map(ref_point);

            model()->screen_pos.x = global_ref_pt.x();
            model()->screen_pos.y = global_ref_pt.y();

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

    if (visible)
    {
        emit gotFocus(); // setFocusPolicy, focusInEvent doesn't work because the CommentView recieves focus to easily

        update();
        view->userinput_update();
    }
}


void CommentView::
        mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
        model()->screen_pos.x = -2;
}


CommentModel* CommentView::
        model()
{
    return dynamic_cast<CommentModel*>(modelp.get());
}


void CommentView::
        closeEvent(QCloseEvent *)
{
    proxy->deleteLater();
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
        model()->scroll_scale *= 1.1;
    else
        model()->scroll_scale /= 1.1;

    update();
}


void CommentView::
        resizeEvent(QResizeEvent *)
{
    model()->window_size = make_uint2( width(), height() );
    keep_pos = true;

    recreatePolygon();
}


bool CommentView::
        testFocus()
{
    return ui->textEdit->hasFocus();
}


void CommentView::
        recreatePolygon()
{
    bool create_thumbnail = model()->thumbnail;

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
        //float s = (ref_point.x()-1)/2;
        float s = std::min(ref_point.x(), 15);
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
    if (model()->thumbnail != v)
    {
        model()->thumbnail = v;

        recreatePolygon();

        setEditFocus(!v);

        emit thumbnailChanged( model()->thumbnail );
    }
}


void CommentView::
        updateText()
{
    model()->html = html();
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


void CommentView::
        setEditFocus(bool focus)
{
    if (focus)
    {
        setFocus(Qt::MouseFocusReason);
        ui->textEdit->setFocus(Qt::MouseFocusReason);
    }
    else
    {
        clearFocus();
        ui->textEdit->clearFocus();
        view->setFocus(Qt::MouseFocusReason);
    }
}


bool CommentView::
        isThumbnail()
{
    return model()->thumbnail;
}


void CommentView::
        updatePosition()
{
    if (!isVisible())
        return;

    bool use_heightmap_value = true;

    // moveEvent can't be used when updating the reference position while moving
    if (!proxy->pos().isNull() || model()->screen_pos.x == -2)
    {
        if (!keep_pos && model()->screen_pos.x == -2)
        {
            QPointF c = proxy->sceneTransform().map(QPointF(ref_point));

            c = view->widget_coordinates( c );

            if (use_heightmap_value)
                model()->pos = view->getHeightmapPos( c );
            else
                model()->pos = view->getPlanePos( c );

            model()->screen_pos.x = -1;
        }

        keep_pos = false;

        move(0,0);
        proxy->scene()->update();
        update();
        view->userinput_update();
    }

    double z;
    QPointF pt;
    if (model()->screen_pos.x >= 0)
    {
        pt.setX( model()->screen_pos.x );
        pt.setY( model()->screen_pos.y );
        z = 6;
    }
    else
    {
        pt = view->getScreenPos( model()->pos, &z, use_heightmap_value );
        lastz = z;
    }
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

    rescale *= model()->scroll_scale;
    //TaskInfo("rescale = %g\tz = %g\tmodel->scroll_scale = %g\tlast_ysize = %g",
    //         rescale, z, model->scroll_scale, view->last_ysize );

    proxy->setTransform(QTransform()
        .translate(pt.x(), pt.y())
        .scale(rescale, rescale)
        .translate( -ref_point.x(), -ref_point.y() )
        );
}

} // namespace Tools
