#include "commentcontroller.h"
#include "commentview.h"

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
    TaskTimer(__FUNCTION__).suppressTiming();
}


CommentView* CommentController::
        createNewComment()
{
    // Create a new comment in the middle of the viewable area
    CommentView* comment = new CommentView(); // view_
    /*comment->qx = view_->_qx;
    comment->qy = view_->_qy;
    comment->qz = view_->_qz;    
*/
    connect(view_, SIGNAL(painting()), comment, SLOT(updatePosition()));

    comment->pos.time = 0/0.f;//view_->_qx;
    comment->pos.scale = view_->_qz;
    comment->view = view_;
    comment->move(0, 0);
    //comment->resize( comment->sizeHint() );

    QGraphicsProxyWidget* proxy = new QGraphicsProxyWidget(0, Qt::Window);
    comment->proxy = proxy;
    proxy->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    proxy->setWidget( comment );
    proxy->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowSystemMenuHint );
    proxy->setCacheMode(QGraphicsItem::ItemCoordinateCache);
    // ZValue is set in commentview
    proxy->setVisible(true);

    view_->addItem( proxy );

    comments_.append( comment );

    return comment;
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
        c->setVisible( active );
}


void CommentController::
        mouseMoveEvent ( QMouseEvent * e )
{
    QPointF c = e->pos();
    float h = height();
    c.setY( h - 1 - c.y() );

    comment_->pos = view_->getHeightmapPos( c );

    view_->userinput_update();

    e->setAccepted(true);
    QWidget::mouseMoveEvent(e);
}

} // namespace Tools
