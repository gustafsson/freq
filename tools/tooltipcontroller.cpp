#include "tooltipcontroller.h"
#include "tooltipview.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

// boost
#include <boost/foreach.hpp>

// Qt
#include <QMouseEvent>

namespace Tools
{

TooltipController::
        TooltipController(RenderView *render_view,
                          CommentController* comments)
            :
            render_view_(render_view),
            comments_(comments),
            current_view_(0)
{
    setupGui();
}


TooltipController::
        ~TooltipController()
{
    BOOST_FOREACH( QPointer<TooltipView>& tv, views_)
    {
        if (tv.data())
            delete tv.data();
    }
    views_.clear();
}


void TooltipController::
        receiveToggleInfoTool(bool active)
{
    render_view_->toolSelector()->setCurrentTool( this, active );
}


void TooltipController::
        mouseReleaseEvent ( QMouseEvent * e )
{
    if( e->button() == Qt::LeftButton)
    {
        infoToolButton.release();
    }
}


void TooltipController::
        mousePressEvent ( QMouseEvent * e )
{
    if(isEnabled()) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton) {
            infoToolButton.press( e->x(), e->y() );

            current_view_ = new TooltipView( this, comments_, render_view_ );
            views_.push_back(current_view_);

            current_model()->max_so_far = -1;
            //if (model()->comment && !model()->comment->model->thumbnail)
                current_model()->comment = 0;
            current_model()->automarking = TooltipModel::AutoMarkerWorking;

            mouseMoveEvent( e );
        }
    }
}


void TooltipController::
        mouseMoveEvent ( QMouseEvent * e )
{
    if (!current_model())
        return;

    if (infoToolButton.isDown())
    {
        bool success=false;
        Heightmap::Position p = render_view_->getPlanePos( e->posF(), &success);
        TaskTimer tt("TooltipController::mouseMoveEvent (%g, %g)", p.time, p.scale);
        if (success)
        {
            current_model()->showToolTip( p );
            render_view_->userinput_update();
        }
    }

    infoToolButton.update(e->x(), e->y());
}


void TooltipController::
        wheelEvent(QWheelEvent *event)
{
    if (!current_model())
        return;

    if (0 < event->delta())
        current_model()->markers++;
    else if(1 < current_model()->markers)
        current_model()->markers--;

    BOOST_ASSERT(current_model()->comment);

    current_model()->automarking = TooltipModel::ManualMarkers;
    current_model()->showToolTip(current_model()->pos );
    render_view_->userinput_update();
}


void TooltipController::
        changeEvent(QEvent *event)
{
    if (event->type() & QEvent::EnabledChange)
        if (!isEnabled())
            emit enabledChanged(isEnabled());
}


void TooltipController::
        setupGui()
{
    Ui::MainWindow* ui = render_view_->model->project()->mainWindow()->getItems();
    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), SLOT(receiveToggleInfoTool(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateInfoTool, SLOT(setChecked(bool)));

    // Close this widget before the OpenGL context is destroyed to perform
    // proper cleanup of resources
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
}


void TooltipController::
        setupView(TooltipView* view_)
{
    // Paint the ellipse when render view paints
    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));
}


TooltipModel* TooltipController::
        current_model()
{
    if (0==current_view_)
        return 0;
    return current_view_->model();
}


} // namespace Tools
