#include "timelinecontroller.h"

#include "support/toolselector.h"
#include "graphicsview.h"
#include "ui_mainwindow.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "timelineview.h"
#include "renderview.h"
#include "sawe/configuration.h"

// Qt
#include <QDockWidget>
#include <QWheelEvent>
#include <QHBoxLayout>
#include <QSettings>

// std
#include <stdio.h>

namespace Tools
{


TimelineController::
        TimelineController( TimelineView* timeline_view )
            :
            model(timeline_view->_render_view->model),
            view(timeline_view),
            dock(0),
            _movingTimeline( 0 )
{
    setupGui();

    setAttribute(Qt::WA_DontShowOnScreen, true);
}


TimelineController::
        ~TimelineController()
{
    TaskInfo("%s", __FUNCTION__);
    if (!dock)
    {
        QSettings().setValue("TimelineController visible", view->tool_selector->parentTool()->isVisible());
    }
}


void TimelineController::
        paintEvent ( QPaintEvent * )
{
    if (!moveButton.isDown())
        return;

    QMouseEvent me(
                QMouseEvent::MouseMove,
                QPoint(moveButton.getLastx (),
                       moveButton.getLasty ()),
                Qt::NoButton, Qt::LeftButton | Qt::RightButton, Qt::NoModifier);

    mousePressEvent(&me);
}


void TimelineController::
        hideTimeline()
{
    if (dock)
        dock->hide();
}


void TimelineController::
        setupGui()
{
    Ui::SaweMainWindow* MainWindow = model->project()->mainWindow();

    bool create_dock_window = Sawe::Configuration::feature("timeline_dock");
    if (create_dock_window)
    {
        view->tool_selector = 0; // explicit

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

        connect(MainWindow->getItems()->actionToggleTimelineWindow, SIGNAL(toggled(bool)), dock, SLOT(setVisible(bool)));
        connect(dock, SIGNAL(visibilityChanged(bool)), MainWindow->getItems()->actionToggleTimelineWindow, SLOT(setChecked(bool)));
    } else {
        view->tool_selector = view->_render_view->graphicsview->toolSelector( 1, model->project()->commandInvoker() );
        view->tool_selector->setCurrentToolCommand( this );

        view->layoutChanged( view->_render_view->graphicsview->layoutDirection());
        connect(view->_render_view->graphicsview, SIGNAL(layoutChanged(QBoxLayout::Direction)),
                view, SLOT(layoutChanged(QBoxLayout::Direction)) );

        embeddedVisibilityChanged(MainWindow->getItems()->actionToggleTimelineWindow->isChecked());
        connect(MainWindow->getItems()->actionToggleTimelineWindow, SIGNAL(toggled(bool)), SLOT(embeddedVisibilityChanged(bool)));
    }

    // Always redraw the timeline whenever the main render view is painted.
    // User input events that changes the state of this widget often need to
    // repaint the main render view also. For them it is enough to issue an
    // update() on the main render view only since this connection makes
    // the timeline updated as well. Some user input events only need to
    // repaint the timeline view.
    connect(view->_render_view, SIGNAL(postPaint()), view, SLOT(update()));
    connect(view->_render_view, SIGNAL(destroying()), view, SLOT(close()));
    connect(view, SIGNAL(hideMe()), SLOT(hideTimeline()));

    if (!dock)
    {
        QSettings settings;
        MainWindow->getItems()->actionToggleTimelineWindow->setChecked(
                settings.value("TimelineController visible").toBool());
    }
}


void TimelineController::
        embeddedVisibilityChanged(bool visible)
{
    EXCEPTION_ASSERT( 0 == dock );

    if (!visible)
    {
        disconnect(view->_render_view, SIGNAL(paintingForeground()), view, SLOT(paintInGraphicsView()));
        disconnect(view->_render_view, SIGNAL(postPaint()), view, SLOT(update()));
    }
    else
    {
        connect(view->_render_view, SIGNAL(paintingForeground()), view, SLOT(paintInGraphicsView()));
        connect(view->_render_view, SIGNAL(postPaint()), view, SLOT(update()));
    }

    if (view->tool_selector)
        view->tool_selector->parentTool()->setVisible(visible);
}


void TimelineController::
        wheelEvent ( QWheelEvent *e )
{
    int x = e->x(), y = e->y();
    float ps = 0.0005f;

    Heightmap::Position p = view->getSpacePos(QPointF(x,y));

    float f = 1.f - ps * e->delta();
    view->_xscale *= f;
    view->_xoffs = p.time-(p.time-view->_xoffs)/f;

    // Only update the timeline, leave the main render view unaffected
    view->redraw();
}


void TimelineController::
        mousePressEvent ( QMouseEvent * e )
{
    int x = e->x(), y = e->y();
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
            // Updates both the timeline and the main render view (the timeline
            // is redrawn whenever the main render view is redrawn).
            view->_render_view->setPosition( current );
        }

        if (moveButton.isDown() && (e->buttons() & Qt::RightButton))
        {
            view->_xoffs -= current.time - prev.time;

            view->redraw();
        }
        break;

   case 2:
        if (e->buttons() & Qt::LeftButton)
        {
            //view->setupCamera( true );
            //moveButton.spacePos(x, y, current[0], current[1]);
            current.time = (current.time - view->_xoffs) * view->_xscale;

            float length = std::max( 1.f, model->project()->length());
            view->_xoffs = current.time - 0.5f*length/view->_xscale;

            view->redraw();
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
