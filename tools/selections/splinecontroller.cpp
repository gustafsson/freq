#include "splinecontroller.h"
#include "splinemodel.h"

// Sonic AWE
#include "tools/selectioncontroller.h"
#include "support/splinefilter.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/renderview.h"

// gpumisc
#include <TaskTimer.h>

// Qt
#include <QMouseEvent>

using namespace Tools;

namespace Tools { namespace Selections
{
    SplineController::
            SplineController(
                    SplineView* view,
                    SelectionController* selection_controller
            )
        :   view_( view ),
            selection_button_( Qt::LeftButton ),
            stop_button_( Qt::RightButton ),
            selection_controller_( selection_controller )
    {
        setupGui();

        setAttribute( Qt::WA_DontShowOnScreen, true );
        setEnabled( false );
    }


    SplineController::
            ~SplineController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void SplineController::
            setupGui()
    {
        Ui::SaweMainWindow* main = selection_controller_->model()->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        // Connect enabled/disable actions,
        // 'enableSquareSelection' sets/unsets this as current tool when
        // the action is checked/unchecked.
        // using actionPolygonSelection instead of actionSplineSelection since polygons are drawn
        connect(ui->actionPolygonSelection, SIGNAL(toggled(bool)), SLOT(enableSplineSelection(bool)));
        connect(this, SIGNAL(enabledChanged(bool)), ui->actionPolygonSelection, SLOT(setChecked(bool)));

        // Paint the ellipse when render view paints
        connect(selection_controller_->render_view(), SIGNAL(painting()), view_, SLOT(draw()));
        // Close this widget before the OpenGL context is destroyed to perform
        // proper cleanup of resources
        connect(selection_controller_->render_view(), SIGNAL(destroying()), SLOT(close()));

        // Add the action as a combo box item in selection controller
        selection_controller_->addComboBoxAction( ui->actionPolygonSelection ) ;
    }


    void SplineController::
            mousePressEvent ( QMouseEvent * e )
    {
        if (e->button() == stop_button_)
        {
            setMouseTracking(false);
            model()->drawing = false;
            model()->v.pop_back();

            selection_controller_->setCurrentSelection( model()->updateFilter() );
        }
        else if (e->button() == selection_button_)
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

            Heightmap::Position click;
            if (Ui::MouseControl::planePos(
                    e->x(), height() - 1 - e->y(),
					click.time, click.scale, r.model->xscale))
            {
                if (!model()->drawing)
                {
                    model()->v.clear();
                    model()->v.push_back( click );
                }
                model()->v.push_back( click );
                model()->drawing = true;
                setMouseTracking(true);
            }
        }

        selection_controller_->render_view()->userinput_update();
    }


    void SplineController::
            mouseReleaseEvent ( QMouseEvent * /*e*/ )
    {
        selection_controller_->render_view()->userinput_update();
    }


    void SplineController::
            mouseMoveEvent ( QMouseEvent * e )
    {

        if (model()->drawing || e->buttons().testFlag( selection_button_ ))
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

            Heightmap::Position click;
            if (Ui::MouseControl::planePos(
                    e->x(), height() - 1 - e->y(),
					click.time, click.scale, r.model->xscale))
            {
                if (!model()->drawing)
                    model()->v.clear();

                if (model()->v.empty())
                {
                    model()->v.push_back( click );
                    model()->drawing = true;
                    setMouseTracking(true);
                }

                model()->v.back() = click;
            }
        }

        selection_controller_->render_view()->userinput_update();
    }


    void SplineController::
            changeEvent ( QEvent * event )
    {
        if (event->type() & QEvent::ParentChange)
        {
            view_->visible = 0!=parent();
        }

        if (event->type() & QEvent::EnabledChange)
        {
            view_->enabled = isEnabled();
            emit enabledChanged(isEnabled());
        }
    }


    void SplineController::
            enableSplineSelection(bool active)
    {
        if (active)
        {
            selection_controller_->setCurrentTool( this, active );
            selection_controller_->setCurrentSelection( model()->updateFilter() );
        }
    }

}} // namespace Tools::Selections
