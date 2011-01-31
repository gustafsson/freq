#include "commentcontroller.h"
#include "commentview.h"
#include "toolfactory.h"

// Sonic AWE
#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

// Qt
#include <QGraphicsProxyWidget>
#include <QMouseEvent>

namespace Tools
{

CommentController::
        CommentController(RenderView* view)
            :   view_(view)
{
    setupGui();

    setEnabled( false );
}


CommentController::
        ~CommentController()
{
    TaskInfo(__FUNCTION__);
}


void CommentController::
        createView( ToolModel* model, Sawe::Project* /*p*/, RenderView* render_view )
{
    CommentModel* cmodel = dynamic_cast<CommentModel*>(model);
    if (0 == cmodel)
        return;

    // Create a new comment in the middle of the viewable area
    CommentView* comment = new CommentView(cmodel);

    connect(render_view, SIGNAL(painting()), comment, SLOT(updatePosition()));

    comment->view = render_view;
    comment->move(0, 0);
    comment->resize( cmodel->window_size.x, cmodel->window_size.y );

    QGraphicsProxyWidget* proxy = new QGraphicsProxyWidget(0, Qt::Window);
    comment->proxy = proxy;
    proxy->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy->setWidget( comment );
    proxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    // ZValue is set in commentview
    proxy->setVisible(true);

    render_view->addItem( proxy );

    comments_.append( comment );
}


void CommentController::
        setComment( Heightmap::Position p, std::string text, QPointer<CommentView>* viewp )
{
    QPointer<CommentView> myview;
    if (!viewp)
        viewp = &myview;
    QPointer<CommentView>& view = *viewp;

    if (!view)
    {
        view = createNewComment();

        view->model->freezed_position = true;
    }

    view->model->pos = p;
    view->setHtml( text );
    view->show();
}


CommentView* CommentController::
        createNewComment()
{
    CommentModel* model = new CommentModel();
    ToolModelP modelp(model);
    view_->model->project()->tools().toolModels.insert( modelp );

    model->pos.time = -FLT_MAX;//view_->model->_qx;
    //model->pos.scale = view_->model->_qz;

    createView(modelp.get(), view_->model->project(), view_ );
    return comments_.back();
}


void CommentController::
        setupGui()
{
    Ui::MainWindow* ui = view_->model->project()->mainWindow()->getItems();

    // Connect enabled/disable actions,
    // 'enableCommentAdder' sets/unsets this as current tool when
    // the action is checked/unchecked.
    connect(ui->actionAddComment, SIGNAL(toggled(bool)), SLOT(enableCommentAdder(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionAddComment, SLOT(setChecked(bool)));

    connect(ui->actionShowComments, SIGNAL(toggled(bool)), SLOT(showComments(bool)));
    showComments(ui->actionShowComments->isChecked());
}


void CommentController::
        changeEvent ( QEvent * event )
{
    if (event->type() & QEvent::EnabledChange)
    {
        if (!isEnabled() && parent())
            dynamic_cast<QWidget*>(parent())->graphicsProxyWidget()->setZValue(-1e30);
        emit enabledChanged(isEnabled());
    }
}


void CommentController::
        enableCommentAdder(bool active)
{
    view_->toolSelector()->setCurrentTool( this, active );

    if (active)
    {
        comment_ = createNewComment();
        setVisible( true );

        dynamic_cast<QWidget*>(parent())->graphicsProxyWidget()->setZValue(1e30);

        setMouseTracking( true );
        connect(comment_, SIGNAL(setCommentControllerEnabled(bool)), SLOT(setEnabled(bool)));
    }
}


void CommentController::
        showComments(bool active)
{
    foreach (CommentView* c, comments_)
        c->setVisible( !active );
}


void CommentController::
        mouseMoveEvent ( QMouseEvent * e )
{
    QPointF c = e->pos();
    float h = height();
    c.setY( h - 1 - c.y() );

    comment_->model->pos = view_->getHeightmapPos( c );

    view_->userinput_update();

    e->setAccepted(true);
    QWidget::mouseMoveEvent(e);
}

} // namespace Tools
