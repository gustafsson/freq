#include "squarecontroller.h"
#include "squaremodel.h"

// Sonic AWE
#include "tools/selectioncontroller.h"
#include "filters/rectangle.h"
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
    SquareController::
            SquareController(
                    SquareView* view,
                    SelectionController* selection_controller
            )
        :   view_( view ),
            selection_button_( Qt::LeftButton ),
            selection_controller_( selection_controller )
    {
        setupGui();

        setAttribute( Qt::WA_DontShowOnScreen, true );
        setEnabled( false );
    }


    SquareController::
            ~SquareController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void SquareController::
            setupGui()
    {
        Ui::SaweMainWindow* main = selection_controller_->model()->project->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        // Connect enabled/disable actions,
        // 'enableSquareSelection' sets/unsets this as current tool when
        // the action is checked/unchecked.
        connect(ui->actionSquareSelection, SIGNAL(toggled(bool)), SLOT(enableSquareSelection(bool)));
        connect(this, SIGNAL(enabledChanged(bool)), ui->actionSquareSelection, SLOT(setChecked(bool)));

        // Paint the ellipse when render view paints
        connect(selection_controller_->render_view(), SIGNAL(painting()), view_, SLOT(draw()));
        // Close this widget before the OpenGL context is destroyed to perform
        // proper cleanup of resources
        connect(selection_controller_->render_view(), SIGNAL(destroying()), SLOT(close()));

        // Add the action as a combo box item in selection controller
        selection_controller_->addComboBoxAction( ui->actionSquareSelection ) ;
    }


    void SquareController::
            mousePressEvent ( QMouseEvent * e )
    {
        if (e->button() == selection_button_)
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

            if (false == Ui::MouseControl::planePos(
                    e->x(), height() - e->y(),
                    selectionStart.time, selectionStart.scale, r.xscale))
            {
                selectionStart.time = -FLT_MAX;
            }
        }

        selection_controller_->render_view()->userinput_update();
    }


    void SquareController::
            mouseReleaseEvent ( QMouseEvent * e )
    {
        if (e->button() == selection_button_)
        {
            model()->updateFilter();

            selection_controller_->setCurrentSelection( model()->filter );
        }

        selection_controller_->render_view()->userinput_update();
    }


    void SquareController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        if (e->buttons().testFlag( selection_button_ ))
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

            Heightmap::Position p;
            if (Ui::MouseControl::planePos(
                    e->x(), height() - e->y(),
                    p.time, p.scale, r.xscale))
            {
                if (-FLT_MAX == selectionStart.time) // TODO test
                {
                    selectionStart = p;
                }

                model()->a = selectionStart;
                model()->b = p;
            }

        }

        selection_controller_->render_view()->userinput_update();
    }


    void SquareController::
            changeEvent ( QEvent * event )
    {
        if (event->type() & QEvent::EnabledChange)
        {
            view_->enabled = isEnabled();
            emit enabledChanged(isEnabled());
        }
    }


    void SquareController::
            enableSquareSelection(bool active)
    {
        selection_controller_->setCurrentTool( this, active );

        if (active)
            selection_controller_->setCurrentSelection( model()->filter );
    }

}} // namespace Tools::Selections
