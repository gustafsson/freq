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
        createView( ToolModelP modelp, ToolRepo* repo, Sawe::Project* /*p*/ )
{
    TooltipModel* model = dynamic_cast<TooltipModel*>(modelp.get());
    if (!model)
        return;

    model->setPtrs( repo->render_view(), this->comments_ );

    setCurrentView( new TooltipView( model, this, repo->render_view() ) );
    setupView(current_view_);
}


void TooltipController::
        setCurrentView(TooltipView* value )
{
    if (views_.end() == std::find(views_.begin(), views_.end(), value))
    {
        current_view_ = 0;
        views_.push_back(value);
    }

    if (value != current_view_)
    {
        current_view_ = value;
        emit tooltipChanged();
    }

    if (value == 0)
    {
        render_view_->toolSelector()->setCurrentTool( this, value != 0 );
        setEnabled( value != 0 );
    }
}


TooltipView* TooltipController::
        current_view()
{
    return current_view_;
}


void TooltipController::
        receiveToggleInfoTool(bool active)
{
    render_view_->toolSelector()->setCurrentTool( this, active );

    emitTooltipChanged();
}


void TooltipController::
        emitTooltipChanged()
{
    emit tooltipChanged();
    render_view_->userinput_update(false);
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

            TooltipModel* model = new TooltipModel();

            model->automarking = TooltipModel::AutoMarkerWorking;

            render_view_->model->project()->tools().addModel( model );

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
            Heightmap::Position o = current_model()->pos();
            bool t = TooltipModel::AutoMarkerWorking == current_model()->automarking;

            current_model()->showToolTip( p );

            t |= current_model()->pos() != o;

            if (t)
                emitTooltipChanged();

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
    current_model()->showToolTip(current_model()->pos() );
    emitTooltipChanged();
    render_view_->userinput_update();
}


void TooltipController::
        changeEvent(QEvent *event)
{
    if (event->type() & QEvent::EnabledChange)
    {
        if (!isEnabled())
            setCurrentView(0);

        emit enabledChanged(isEnabled());
    }
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
    connect(view_, SIGNAL(tooltipChanged()), SLOT(emitTooltipChanged()));
    connect(view_, SIGNAL(destroyed()), SLOT(emitTooltipChanged()));
}


TooltipModel* TooltipController::
        current_model()
{
    if (0==current_view_)
        return 0;
    return current_view_->model();
}


} // namespace Tools
