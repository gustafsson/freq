#include "playbackmarkerscontroller.h"
#include "renderview.h"
#include "rendermodel.h"
#include "heightmap/renderer.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include <QMouseEvent>

namespace Tools {

PlaybackMarkersController::
        PlaybackMarkersController( PlaybackMarkersView* view, RenderView* render_view )
    :
    vicinity_( 10 ),
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
        render_view_->toolSelector()->setCurrentTool( this, active );

    setMouseTracking(active);

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        receivePreviousMarker()
{
    PlaybackMarkersModel::Markers::iterator itr = model()->currentMarker();

    if (itr != model()->markers().begin())
    {
        itr--;
        model()->setCurrentMaker( itr );
    }

    render_view_->setPosition( *itr, render_view_->model->_qz );
    render_view_->userinput_update();
}


void PlaybackMarkersController::
        receiveNextMarker()
{
    PlaybackMarkersModel::Markers::iterator itr = model()->currentMarker();

    PlaybackMarkersModel::Markers::iterator next_itr = itr;
    next_itr++;

    float pos = *next_itr;
    if (next_itr != model()->markers().end())
    {
        model()->setCurrentMaker( next_itr );
    }
    else
    {
        pos = render_view_->model->project()->worker.length();
    }

    render_view_->setPosition( pos, render_view_->model->_qz );
    render_view_->userinput_update();
}


void PlaybackMarkersController::
        mousePressEvent ( QMouseEvent * e )
{
    e->accept();

    RenderView &r = *render_view_;
    bool success;
    Heightmap::Position click = r.getPlanePos( QPointF(e->x(), height() - 1 - e->y()), &success);
    if (!success)
        // Meaningless click, ignore
        return;

    PlaybackMarkersModel::Markers::iterator itr = model()->findMaker( click.time );
    if (itr == model()->markers().end())
    {
        if (e->button() == Qt::LeftButton)
        {
            // No markers created, create one
            model()->addMarker( click.time );
        }
    }
    else
    {
        // Find out the distance to the nearest marker.
        Heightmap::Position marker_pos( *itr, click.scale );
        QPointF pt = r.getScreenPos( marker_pos, 0 );

        pt -= e->posF();
        float distance = std::sqrt( pt.x()*pt.x() + pt.y()*pt.y() );
        if (distance < vicinity_)
        {
            model()->setCurrentMaker( itr );

            if (e->button() == Qt::RightButton)
                model()->removeMarker( model()->currentMarker() );
        }
        else
        {
            if (e->button() == Qt::LeftButton)
            {
                // Click in-between markers, add a new marker
                model()->addMarker( click.time );
            }
        }
    }

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        mouseMoveEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *render_view_;
    bool success;
    Heightmap::Position click = r.getPlanePos( QPointF(e->x(), height() - 1 - e->y()), &success);
    if (!success)
        return;

    PlaybackMarkersModel::Markers::iterator itr = model()->findMaker( click.time );
    if (itr == model()->markers().end())
    {
        // No markers created
        view_->setHighlightMarker( click.time, true );
    }
    else
    {
        // Find out the distance to the nearest marker.
        Heightmap::Position marker_pos( *itr, click.scale );
        QPointF pt = r.getScreenPos( marker_pos, 0 );

        pt -= e->posF();
        float distance = std::sqrt( pt.x()*pt.x() + pt.y()*pt.y() );
        if (distance < vicinity_)
            view_->setHighlightMarker( *itr, false );
        else
            view_->setHighlightMarker( click.time, true );
    }

    render_view_->userinput_update();
}


void PlaybackMarkersController::
        changeEvent(QEvent *event)
{
    if (event->type() & QEvent::EnabledChange)
    {
        if (!isEnabled())
            view_->setHighlightMarker( -1, false );

        view_->enabled = isEnabled();
        emit enabledChanged(isEnabled());
    }
}


void PlaybackMarkersController::
        setupGui()
{
    Ui::SaweMainWindow* main = render_view_->model->project()->mainWindow();
    Ui::MainWindow* ui = main->getItems();

    // Connect enabled/disable actions,
    connect(ui->actionSetPlayMarker, SIGNAL(toggled(bool)), SLOT(enableMarkerTool(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionSetPlayMarker, SLOT(setChecked(bool)));
    connect(ui->actionJumpForward, SIGNAL(triggered()), SLOT(receiveNextMarker()));
    connect(ui->actionJumpBackward, SIGNAL(triggered()), SLOT(receivePreviousMarker()));


    // Paint when render view paints
    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));
    // Close this widget before the OpenGL context is destroyed to perform
    // proper cleanup of resources
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
}

} // namespace Tools
