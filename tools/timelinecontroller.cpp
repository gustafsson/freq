#include "timelinecontroller.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "timelineview.h"
#include "renderview.h"

// Qt
#include <QDockWidget>
#include <QWheelEvent>

namespace Tools
{


TimelineController::
        TimelineController( TimelineView* timeline_view )
            :
            model(timeline_view->_render_view->model),
            view(timeline_view),
            _movingTimeline( 0 )
{

}


void TimelineController::
        setupGui()
{
    Ui::SaweMainWindow* MainWindow = dynamic_cast<Ui::SaweMainWindow*>(model->project->mainWindow());
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTimeline"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    dock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dock->setEnabled(true);
    dock->setAutoFillBackground(true);
    dock->setWidget(this);
    dock->setWindowTitle("Timeline");
    dock->show();
    MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dock);

    connect(view->_render_view, SIGNAL(paintedView()), view, SLOT(update()));
}


/*void TimelineView::
        put(Signal::pBuffer , Signal::pOperation )
{
    update();
}


void TimelineView::
        add_expected_samples( const Signal::Intervals& )
{
    update();
}*/


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

    view->_render_view->update();
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

    if (e->buttons() & Qt::LeftButton)
    {
        if (0 == _movingTimeline)
        {
            if (current[1]>=0)  _movingTimeline = 1;
            else                _movingTimeline = 2;
        }

        switch ( _movingTimeline )
        {
        case 1:
            view->_render_view->setPosition( current[0], current[1] );
            break;
        case 2:
            {
                view->setupCamera( true );
                moveButton.spacePos(x, y, current[0], current[1]);

                float length = std::max( 1.f, model->project->worker.source()->length());
                view->_xoffs = current[0] - 0.5f*length/view->_xscale;
            }
            break;
        }
    }

    if (moveButton.isDown() && (e->buttons() & Qt::RightButton))
    {
        view->_xoffs -= current[0] - prev[0];
    }

    moveButton.press( x, y );
    view->_render_view->update();
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
