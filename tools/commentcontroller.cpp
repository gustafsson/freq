#include "commentcontroller.h"
#include "commentview.h"
#include "toolfactory.h"

// Sonic AWE
#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "graphicsview.h"

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
        createView( ToolModelP model, ToolRepo* repo, Sawe::Project* /*p*/ )
{
    CommentModel* cmodel = dynamic_cast<CommentModel*>(model.get());
    if (0 == cmodel)
        return;

    // Create a new comment in the middle of the viewable area
    CommentView* comment = new CommentView(model);

    connect(repo->render_view(), SIGNAL(painting()), comment, SLOT(updatePosition()));

    comment->view = repo->render_view();
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

    repo->render_view()->addItem( proxy );

    comments_.append( comment );
}


CommentView* CommentController::
        findView( ToolModelP model )
{
    foreach (const QPointer<CommentView>& c, comments_)
        if (c->modelp == model)
            return c;
    return 0;
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

        view->model()->freezed_position = true;
    }

    view->model()->pos = p;
    view->setHtml( text );
    view->show();
}


CommentView* CommentController::
        createNewComment()
{
    CommentModel* model = new CommentModel();

    model->pos.time = -FLT_MAX;//view_->model->_qx;
    //model->pos.scale = view_->model->_qz;

    // addModel calls createView
    view_->model->project()->tools().addModel( model );

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
        view_->toolSelector()->setCurrentTool( this, isEnabled() );

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
        comment_->model()->move_on_hover = true;
        setVisible( true );

        setMouseTracking( true );
        mouseMoveEvent(new QMouseEvent(
                QEvent::MouseMove,
                mapFromGlobal(view_->graphicsview->mapFromGlobal( QCursor::pos())),
                Qt::NoButton,
                Qt::MouseButtons(),
                Qt::KeyboardModifiers()));
    }
    else
    {
        if (comment_ && comment_->model()->move_on_hover)
        {
            comment_->proxy->deleteLater();
            comment_ = 0;
            setVisible( false );
        }
    }
}


void CommentController::
        showComments(bool active)
{
    foreach (const QPointer<CommentView>& c, comments_)
        if (c)
            c->setVisible( !active );
}


void CommentController::
        mouseMoveEvent ( QMouseEvent * e )
{
    bool use_heightmap_value = true;

    if (use_heightmap_value)
        comment_->model()->pos = view_->getHeightmapPos( e->posF() );
    else
        comment_->model()->pos = view_->getPlanePos( e->posF() );
    QPointF window_coordinates = view_->window_coordinates( e->posF() );
    comment_->model()->screen_pos.x = window_coordinates.x();
    comment_->model()->screen_pos.y = window_coordinates.y();

    comment_->setVisible(true);

    view_->userinput_update();

    e->setAccepted(true);
    QWidget::mouseMoveEvent(e);
}


void CommentController::
        mousePressEvent( QMouseEvent * e )
{
    setEnabled( false );
    if (comment_)
    {
        comment_->model()->screen_pos.x = -2;
        comment_->setEditFocus(true);
    }

    e->setAccepted(true);
    QWidget::mousePressEvent(e);
}

} // namespace Tools
