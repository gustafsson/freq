#include "timelinecontroller.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "timelineview.h"
#include "renderview.h"

// Gpumisc
#include <cuda_vector_types_op.h>

// Qt
#include <QDockWidget>
#include <QWheelEvent>
#include <QHBoxLayout>

// std
#include <stdio.h>

namespace Tools
{


TimelineController::
        TimelineController( TimelineView* timeline_view )
            :
            model(timeline_view->_render_view->model),
            view(timeline_view),
            _movingTimeline( 0 )
{
    setupGui();

    setAttribute(Qt::WA_DontShowOnScreen, true);
}


TimelineController::
        ~TimelineController()
{
    TaskInfo("%s", __FUNCTION__);
}


void TimelineController::
        hideTimeline()
{
    dock->hide();
}


void TimelineController::
        setupGui()
{
    Ui::SaweMainWindow* MainWindow = model->project()->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTimeline"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    dock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dock->setEnabled(true);
    dock->setAutoFillBackground(true);
    dock->setWidget(view);
    dock->setWindowTitle("Timeline");
    dock->show();

    MainWindow->addDockWidget(Qt::BottomDockWidgetArea, dock);

    view->setLayout( new QHBoxLayout );
    view->layout()->setMargin( 0 );
    view->layout()->addWidget( this );

    // Always redraw the timeline whenever the main render view is painted.
    // User input events that changes the state of this widget often need to
    // repaint the main render view also. For them it is enough to issue an
    // update() on the main render view only since this connection makes
    // the timeline updated as well. Some user input events only need to
    // repaint the timeline view.
    connect(view->_render_view, SIGNAL(postPaint()), view, SLOT(update()));
    connect(view->_render_view, SIGNAL(destroying()), view, SLOT(close()));
    connect(view, SIGNAL(hideMe()), SLOT(hideTimeline()));
}


void TimelineController::
        wheelEvent ( QWheelEvent *e )
{
    int x = e->x(), y = height() - 1 - e->y();
    float ps = 0.0005f;

    Heightmap::Position p = view->getSpacePos(QPointF(x,y));

    float f = 1.f - ps * e->delta();
    view->_xscale *= f;
    view->_xoffs = p.time-(p.time-view->_xoffs)/f;

    // Only update the timeline, leave the main render view unaffected
    view->userinput_update();
}


void TimelineController::
        mousePressEvent ( QMouseEvent * e )
{
    int x = e->x(), y = height() - 1 - e->y();
    Heightmap::Position prev = view->getSpacePos(QPointF(moveButton.getLastx(), moveButton.getLasty()));
    Heightmap::Position current = view->getSpacePos(QPointF(x, y));

    if (0 == _movingTimeline)
    {
        if (current.scale>=0)   _movingTimeline = 1;
        else                    _movingTimeline = 2;
    }

    switch ( _movingTimeline )
    {
    case 1:
        if (e->buttons() & Qt::LeftButton)
        {
            view->_render_view->setPosition( current.time, current.scale );

            // Update both the timeline and the main render view (the timeline
            // is redrawn whenever the main render view is redrawn).
            view->_render_view->userinput_update();
        }

        if (moveButton.isDown() && (e->buttons() & Qt::RightButton))
        {
            view->_xoffs -= current.time - prev.time;

            // Only update the timeline, leave the main render view unaffected
            view->userinput_update();
        }
        break;

   case 2:
        if (e->buttons() & Qt::LeftButton)
        {
            //view->setupCamera( true );
            //moveButton.spacePos(x, y, current[0], current[1]);
            current.time = (current.time - view->_xoffs) * view->_xscale;

            float length = max1( 1.f, model->project()->worker.source()->length());
            view->_xoffs = current.time - 0.5f*length/view->_xscale;

            // Only update the timeline, leave the main render view unaffected
            view->userinput_update();
        }
        break;
    }

    moveButton.press( x, y );
}


void TimelineController::
        mouseReleaseEvent ( QMouseEvent * e)
{
    if (0 == (e->buttons() & Qt::LeftButton)) {
        _movingTimeline = 0;
    }
    moveButton.release();
}


void TimelineController::
        mouseMoveEvent ( QMouseEvent * e )
{
    mousePressEvent(e);
}

}
