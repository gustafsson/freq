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
    CommentView* comment = new CommentView(); // view_
    /*comment->qx = view_->_qx;
    comment->qy = view_->_qy;
    comment->qz = view_->_qz;    
*/
    connect(view_, SIGNAL(painting()), comment, SLOT(updatePosition()));

    comment->pos.time = view_->_qx;
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
    //proxy->setZValue(0); // Make sure the dialog is drawn on top of all other (OpenGL) items
    proxy->setVisible(true);

    view_->addItem( proxy );
}


void CommentController::
        setupGui()
{
    Ui::MainWindow* ui = view_->model->project()->mainWindow()->getItems();
    connect(ui->actionAddComment, SIGNAL(triggered()), this, SLOT(receiveAddComment()));
}


} // namespace Tools
