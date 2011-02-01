#include "playbackmarkerscontroller.h"
#include "renderview.h"
#include "rendermodel.h"
#include "heightmap/renderer.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include <QMouseEvent>

namespace Tools {

PlaybackMarkersController::PlaybackMarkersController( PlaybackMarkersView* view, RenderView* render_view )
    :
    state_(Inactive),
    render_view_(render_view),
    view_(view)
{
    setupGui();

    setAttribute( Qt::WA_DontShowOnScreen, true );
    setEnabled( false );
}


void PlaybackMarkersController::
        enableMarkerTool(bool active)
{
    if (active)
        _view->toolSelector()->setCurrentTool( this, active );

    setMouseTracking(active);

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        removeCurrentMarker()
{
    model()->removeMarker( model()->currentMarker() );

    render_view_->userinput_update( false );
}


void PlaybackMarkersController::
        selectCurrentMarker(bool active)
{
    if (active)
        _view->toolSelector()->setCurrentTool( this, active );

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        mousePressEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *selection_controller_->render_view();
    bool success;
    Heightmap::Position click = r.getPlanePos( QPointF(e->x(), height() - 1 - e->y()), &success);
    if (!success)
        // Meaningless click, ignore
        return;

    Markers::iterator itr = model()->findMaker( click.time );
    if (itr == model()->markers().end())
    {
        // No markers created, create one
        model()->addMarker( click.time );
        return;
    }

    // Find out the distance to the nearest marker.
    Heightmap::Position marker_pos( *itr, click.scale );
    QPointF pt = r.getScreenPos( marker_pos );

    pt -= e->posF();
    float distance = std::sqrt( pt.x()*pt.x() + pt.y()*pt.y() );
    if (distance < vicinity)
    {
        model()->setCurrentMaker( itr );
        return;
    }

    // Click in-between markers, add a new marker
    model()->addMarker( click.time );

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        mouseMoveEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *selection_controller_->render_view();
    bool success;
    Heightmap::Position click = r.getPlanePos( QPointF(e->x(), height() - 1 - e->y()), &success);
    if (!success)
        return;

    Markers::iterator itr = model()->findMaker( click.time );
    if (itr == model()->markers().end())
    {
        // No markers created
        return;
    }

    // Find out the distance to the nearest marker.
    Heightmap::Position marker_pos( *itr, click.scale );
    QPointF pt = r.getScreenPos( marker_pos );

    pt -= e->posF();
    float distance = std::sqrt( pt.x()*pt.x() + pt.y()*pt.y() );
    if (distance < vicinity)
    {
        view_->setHighlightMarker( itr );
        return;
    }

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        changeEvent(QEvent *)
{
    if (event->type() & QEvent::EnabledChange)
    {
        if (!isEnabled())
            view_->setAddingMarker( -1 );

        view_->enabled = isEnabled();
    }
}


void PlaybackMarkersController::
        setupGui()
{
    Ui::SaweMainWindow* main = selection_controller_->model()->project()->mainWindow();
    Ui::MainWindow* ui = main->getItems();

    // Connect enabled/disable actions,
    connect(ui->actionSetPlayMarker, SIGNAL(toggled(bool)), SLOT(enableMarkerTool(bool)));

    // Paint when render view paints
    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));
    // Close this widget before the OpenGL context is destroyed to perform
    // proper cleanup of resources
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
}

} // namespace Tools
