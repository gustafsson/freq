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
#include <QScrollBar>

namespace Tools {

CommentView::CommentView(ToolModelP modelp, RenderView* render_view, QWidget *parent) :
    QWidget(parent),
    view( render_view ),
    modelp(modelp),
    ui(new Ui::CommentView),
    proxy( 0 ),
    keep_pos(false),
    z_hidden(false),
    lastz(6)
{
    ui->setupUi(this);

    BOOST_ASSERT( dynamic_cast<CommentModel*>(modelp.get() ));

    this->setPalette(QPalette(QPalette::Window, QColor(255,0,0,0)));

    QAction *closeAction = new QAction(tr("D&elete"), this);
    connect(closeAction, SIGNAL(triggered()), SLOT(close()));

    QAction *hideAction = new QAction(tr("T&humbnail"), this);
    hideAction->setCheckable(true);
    connect(hideAction, SIGNAL(toggled(bool)), SLOT(thumbnail(bool)));
    connect(this, SIGNAL(thumbnailChanged(bool)), hideAction, SLOT(setChecked(bool)));

	connect(ui->textEdit, SIGNAL(textChanged()), SLOT(updateText()));
    addAction(closeAction);
    addAction(hideAction);
    setMouseTracking( true );
    setHtml(model()->html);
    connect(ui->textEdit, SIGNAL(selectionChanged()), SLOT(recreatePolygon()));

    connect(render_view, SIGNAL(painting()), SLOT(updatePosition()));

    proxy = new QGraphicsProxyWidget(0, Qt::Window);
    proxy->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy->setWidget( this );
    proxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    // ZValue is set in CommentView::updatePosition()
    proxy->setVisible(true);
    render_view->addItem( proxy );

    move(0, 0);
    resize( model()->window_size[0], model()->window_size[1] );
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

    // grow if needed to display all contents, but only grow as long as this widget is smaller than maxGrowSize
    {
        QSize maxGrowSize(500, 200);

        // disable word wrap to grow horizontally as far as needed, trying to avoid word wrap
        QTextOption::WrapMode prevWrapMode = ui->textEdit->wordWrapMode();
        ui->textEdit->setWordWrapMode( QTextOption::NoWrap );

        // disable any vertical scrollbar while growing horizontally
        ui->textEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

        // grow horizontally
        for (QScrollBar* hori = ui->textEdit->horizontalScrollBar(); hori && hori->isVisible() && width() < maxGrowSize.width(); resize( width() + 1, height() )) {}

        if (maxGrowSize.width() <= width())
        {
            // we might need word wrap, and/or a horizontal scrollbar
            ui->textEdit->setWordWrapMode( prevWrapMode );
        }
        else
        {
            // disable any horizontal scrollbar while growing vertically if it should be wide enough if it wasn't for the vertical scrollbar
            ui->textEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        }

        // grow veritcally
        ui->textEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        for (QScrollBar* vert = ui->textEdit->verticalScrollBar(); vert && vert->isVisible() && height() < maxGrowSize.height(); resize( width(), height() + 1 )) {}

        // we still want scrollbars if they are needed despite growing the size
        ui->textEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    }
}


void CommentView::
        mousePressEvent(QMouseEvent *event)
{
    // any click outside the mask is discarded
    if (!maskedRegion.contains( event->pos() ))
    {
        setEditFocus( false );
        event->setAccepted( false );
        return;
    }

    // any click sets edit focus to the text widget
    if (!testFocus())
    {
        setEditFocus(true);
    }

    // click with the left mouse button initializes a move or resize
    if (event->buttons() & Qt::LeftButton)
    {
        QPoint gp = proxy->sceneTransform().map(event->pos());

        if (event->modifiers() == 0)
        {
            dragPosition = event->pos();
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
    if (!maskedRegion.contains( event->pos() ))
    {
        event->setAccepted( false );
        return;
    }

    thumbnail( !model()->thumbnail );
}


void CommentView::
        mouseMoveEvent(QMouseEvent *event)
{
    bool visible = maskedRegion.contains( event->pos() );
    setContextMenuPolicy( visible ? Qt::ActionsContextMenu : Qt::NoContextMenu);

    bool moving = false;
    bool resizing = false;

    if (event->buttons() & Qt::LeftButton)
    {
        moving |= event->modifiers() == 0 && !model()->freezed_position;
        resizing |= event->modifiers().testFlag(Qt::ControlModifier);
    }


    if (moving || resizing)
    {
        QPoint gp = proxy->sceneTransform().map(event->pos());

        if (moving)
        {
            move(event->pos() - dragPosition);
            QPoint global_ref_pt = proxy->sceneTransform().map(ref_point);

            model()->screen_pos[0] = global_ref_pt.x();
            model()->screen_pos[1] = global_ref_pt.y();

            event->accept();
        }
        else if (resizing)
        {
            QPoint sz = QPoint(gp.x(),-gp.y()) - resizePosition;
            resize(sz.x(), sz.y());
            event->accept();
        }
        resizePosition = -QPoint(width(), height()) + QPoint(gp.x(), -gp.y());

        view->model->project()->setModified();
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
        model()->screen_pos[0] = UpdateModelPositionFromScreen;
}


QGraphicsProxyWidget* CommentView::
        getProxy()
{
    return proxy;
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
    if (!maskedRegion.contains( e->pos() ))
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
    model()->window_size = tvector<2, unsigned>( width(), height() );
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
    ref_point.setX( std::max(ref_point.x(), 25));

    if (!create_thumbnail)
    {
        ui->textEdit->setVisible( true );
        poly.clear();
        poly.push_back(b - 2*h0);
        poly.push_back(b + 2*x0);
        //poly.push_back(b + 0.1f*y + 0.05f*x);
        //poly.push_back(ref_point);
        //poly.push_back(b + 0.1f*y + 0.15f*x);
        poly.push_back(b + 0.1f*y + ( ref_point.x() - 0.9f*y.y()) *x0);
        poly.push_back(ref_point);
        poly.push_back(b + 0.1f*y + ( ref_point.x() + 0.9f*y.y())*x0);
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
        float s = std::min(ref_point.x(), 15);
        poly.push_back(ref_point);
        poly.push_back(ref_point - s*h0 - s*x0);
        poly.push_back(ref_point - s*h0 + s*x0);
    }


    maskedRegion = r;
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

    //setMask(maskedRegion);

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
    view->model->project()->setModified();
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
    bool use_heightmap_value = true;

    // moveEvent can't be used when updating the reference position while moving
    if (!proxy->pos().isNull() || model()->screen_pos[0] == UpdateModelPositionFromScreen)
    {
        if (!keep_pos && model()->screen_pos[0] == UpdateModelPositionFromScreen)
        {
            QPointF c = proxy->sceneTransform().map(QPointF(ref_point));

            c = view->widget_coordinates( c );

            if (use_heightmap_value)
                model()->pos = view->getHeightmapPos( c );
            else
                model()->pos = view->getPlanePos( c );

            model()->screen_pos[0] = UpdateScreenPositionFromWorld;
        }

        keep_pos = false;

        move(0,0);
        proxy->scene()->update();
        update();
        view->userinput_update();
    }

    double z;
    QPointF pt;
    if (model()->screen_pos[0] != UpdateScreenPositionFromWorld && model()->screen_pos[0] != UpdateModelPositionFromScreen)
    {
        pt.setX( model()->screen_pos[0] );
        pt.setY( model()->screen_pos[1] );
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
