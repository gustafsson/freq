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

    CommentView* comment = new CommentView(model, repo->render_view());

    comments_.append( comment );
}


CommentView* CommentController::
        findView( ToolModelP model )
{
    foreach (const QPointer<CommentView>& c, comments_)
        if (c)
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
        view_->graphicsview->setToolFocus( isEnabled() );

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
        if (comment_)
        {
            // didn't place new comment before tool was disabled
            comment_->getProxy()->deleteLater();
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
    BOOST_ASSERT( comment_ );

    bool use_heightmap_value = true;

    if (use_heightmap_value)
        comment_->model()->pos = view_->getHeightmapPos( e->posF() );
    else
        comment_->model()->pos = view_->getPlanePos( e->posF() );
    QPointF window_coordinates = view_->window_coordinates( e->posF() );
    comment_->model()->screen_pos.x = window_coordinates.x();
    comment_->model()->screen_pos.y = window_coordinates.y();

    view_->userinput_update();

    e->setAccepted(true);
    QWidget::mouseMoveEvent(e);
}


void CommentController::
        mousePressEvent( QMouseEvent * e )
{
    if (comment_)
    {
        comment_->model()->screen_pos.x = UpdateModelPositionFromScreen;
        comment_->setEditFocus(true);
        comment_ = 0;
    }

    setEnabled( false );

    e->setAccepted(true);
    QWidget::mousePressEvent(e);
}

} // namespace Tools
