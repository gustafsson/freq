#include "commentcontroller.h"
#include "commentview.h"

// Sonic AWE
#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

// Qt
#include <QGraphicsProxyWidget>

namespace Tools
{

CommentController::
        CommentController(RenderView* view)
            :   view_(view)
{
    setupGui();
}


CommentController::
        ~CommentController()
{

}


void CommentController::
        receiveAddComment()
{
    // Create a new comment in the middle of the viewable area
    connect(view_, SIGNAL(painting()), this, SLOT(receiveAddComment()));

    CommentView* comment = new CommentView(view_);
    comment->qx = view_->_qx;
    comment->qy = view_->_qy;
    comment->qz = view_->_qz;

    QGraphicsProxyWidget* proxy = new QGraphicsProxyWidget(0, Qt::Window);
    proxy->setWidget( comment );

    view_->addItem( proxy );
}


void CommentController::
        setupGui()
{
    Ui::MainWindow* ui = view_->model->project()->mainWindow()->getItems();
    connect(ui->actionAddComment, SIGNAL(triggered()), this, SLOT(receiveAddComment()));
}


} // namespace Tools
