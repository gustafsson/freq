#include "commentcontroller.h"
#include "commentview.h"
#include "toolfactory.h"

// Sonic AWE
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "graphicsview.h"
#include "tools/support/renderviewinfo.h"

// Qt
#include <QGraphicsProxyWidget>
#include <QMouseEvent>

namespace Tools
{

using Support::RenderViewInfo;

CommentController::
        CommentController(QGraphicsScene* graphicsscene, RenderView* view, Sawe::Project* project, Support::ToolSelector* tool_selector, GraphicsView* graphicsview)
            :   graphicsscene_(graphicsscene),
                view_(view),
                project_(project),
                tool_selector_(tool_selector),
                graphicsview_(graphicsview),
                comment_(0)
{
    setEnabled( false );

    setupGui();
}


CommentController::
        ~CommentController()
{
    TaskInfo(__FUNCTION__);
}


void CommentController::
        createView( ToolModelP model, ToolRepo* repo, Sawe::Project* p )
{
    CommentModel* cmodel = dynamic_cast<CommentModel*>(model.get());
    if (0 == cmodel)
        return;

    CommentView* comment = new CommentView(model, graphicsscene_, repo->render_view(), p);

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
        view->resize( 1, 1 ); // grow with setHtml instead
    }

    view->model()->pos = p;

    // keep in sync with CommentView::updatePosition()
    view->model()->scroll_scale = 0.5f*sqrt(-view_->model->camera->p[2]);

    view->setHtml( text );
    view->show();
}


CommentView* CommentController::
        createNewComment()
{
    CommentModel* model = new CommentModel();

    model->pos.time = -FLT_MAX;//view_->model->camera.q[0];
    //model->pos.scale = view_->model->camera.q[2];

    // addModel calls createView
    project_->tools().addModel( model );

    return comments_.back();
}


void CommentController::
        setupGui()
{
    Ui::MainWindow* ui = project_->mainWindow()->getItems();

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
    if (event->type() == QEvent::EnabledChange)
        if (!isEnabled())
            emit enabledChanged( isEnabled() );
}


void CommentController::
        enableCommentAdder(bool active)
{
    tool_selector_->setCurrentTool( this, active );
    graphicsview_->setToolFocus( active );

    if (active)
    {
        comment_ = createNewComment();
        setVisible( true );

        setMouseTracking( true );
        mouseMoveEvent(new QMouseEvent(
                QEvent::MouseMove,
                mapFromGlobal(graphicsview_->mapFromGlobal( QCursor::pos())),
                Qt::NoButton,
                Qt::MouseButtons(),
                Qt::KeyboardModifiers()));
    }
    else
    {
        if (comment_)
        {
            // didn't place new comment before tool was disabled
            QGraphicsProxyWidget* proxy = comment_->getProxy();
            proxy->deleteLater();
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
    EXCEPTION_ASSERT( comment_ );

    bool use_heightmap_value = true;

    if (use_heightmap_value)
        comment_->model()->pos = RenderViewInfo(view_->model).getHeightmapPos( e->localPos() );
    else
        comment_->model()->pos = RenderViewInfo(view_->model).getPlanePos( e->localPos() );

    QPointF window_coordinates = RenderViewInfo(view_->model).window_coordinates( e->localPos() );
    comment_->model()->screen_pos[0] = window_coordinates.x();
    comment_->model()->screen_pos[1] = window_coordinates.y();

    view_->redraw();

    e->setAccepted(true);
    QWidget::mouseMoveEvent(e);
}


void CommentController::
        mousePressEvent( QMouseEvent * e )
{
    if (comment_)
    {
        mouseMoveEvent( e );

        comment_->model()->screen_pos[0] = UpdateModelPositionFromScreen;

        // keep in sync with CommentView::updatePosition()
        comment_->model()->scroll_scale = 0.5f*sqrt(-view_->model->camera->p[2]);

        comment_->setEditFocus(true);
        comment_ = 0;
    }

    setEnabled( false );

    e->setAccepted(true);
    QWidget::mousePressEvent(e);
}

} // namespace Tools
