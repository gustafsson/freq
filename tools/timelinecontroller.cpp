#include "timelinecontroller.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "timelineview.h"
#include "renderview.h"

// Qt
#include <QDockWidget>
#include <QWheelEvent>
#include <QHBoxLayout>

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
    connect(view->_render_view, SIGNAL(painting()), view, SLOT(getLengthNow()) );
}


void TimelineController::
        wheelEvent ( QWheelEvent *e )
{
    view->makeCurrent();
    view->setupCamera();

    int x = e->x(), y = height() - e->y();
    float ps = 0.0005f;

    GLvector current;
    moveButton.spacePos(x, y, current[0], current[1]);

    float f = 1.f - ps * e->delta();
    view->_xscale *= f;

    view->setupCamera();

    GLvector newPos;
    moveButton.spacePos(x, y, newPos[0], newPos[1]);

    //_xoffs -= current[0]/prevscale*_xscale-newPos[0];
    //_xoffs = current[0] - _xscale*(x/(float)width());
    view->_xoffs -= newPos[0]-current[0];

    view->setupCamera();

    GLvector newPos2;
    moveButton.spacePos(x, y, newPos2[0], newPos2[1]);

    // float tg = _oldoffs + x * prevscale;
    // float tg2 = _newoffs + x/(float)width() * _xscale;
    //_xoffs -= x/(float)width() * (prevscale-_xscale);

    if (0) printf("[%d, %d] -> [%g, %g, %g] -> (%g, %g)\n",
           x, y,
           current[0], newPos[0], newPos2[0],
           view->_xscale, view->_xoffs);

    // Only update the timeline, leave the main render view unaffected
    view->userinput_update();
}


void TimelineController::
        mousePressEvent ( QMouseEvent * e )
{
    view->makeCurrent();
    view->setupCamera();

    int x = e->x(), y = height() - e->y();

    GLvector prev;
    moveButton.spacePos(prev[0], prev[1]);

    GLvector current;
    moveButton.spacePos(x, y, current[0], current[1]);

    if (0 == _movingTimeline)
    {
        if (current[1]>=0)  _movingTimeline = 1;
        else                _movingTimeline = 2;
    }

    switch ( _movingTimeline )
    {
    case 1:
        if (e->buttons() & Qt::LeftButton)
        {
            view->_render_view->setPosition( current[0], current[1] );

            // Update both the timeline and the main render view (the timeline
            // is redrawn whenever the main render view is redrawn).
            view->_render_view->userinput_update();
        }

        if (moveButton.isDown() && (e->buttons() & Qt::RightButton))
        {
            view->_xoffs -= current[0] - prev[0];

            // Only update the timeline, leave the main render view unaffected
            view->userinput_update();
        }
        break;

   case 2:
        if (e->buttons() & Qt::LeftButton)
        {
            view->setupCamera( true );
            moveButton.spacePos(x, y, current[0], current[1]);

            float length = std::max( 1.f, model->project()->worker.source()->length());
            view->_xoffs = current[0] - 0.5f*length/view->_xscale;

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
