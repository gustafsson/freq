#include "peakcontroller.h"
#include "peakmodel.h"

// Sonic AWE
#include "tools/selectioncontroller.h"
#include "support/peakfilter.h"
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
/*    PeakController::
            PeakController(
                    PeakView* view,
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


    PeakController::
            ~PeakController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void PeakController::
            setupGui()
    {
        Ui::SaweMainWindow* main = selection_controller_->model()->project->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        // Connect enabled/disable actions,
        // 'enablePeakSelection' sets/unsets this as current tool when
        // the action is checked/unchecked.
        connect(ui->actionPeakSelection, SIGNAL(toggled(bool)), SLOT(enablePeakSelection(bool)));
        connect(this, SIGNAL(enabledChanged(bool)), ui->actionPeakSelection, SLOT(setChecked(bool)));

        // Paint the ellipse when render view paints
        connect(selection_controller_->render_view(), SIGNAL(painting()), view_, SLOT(draw()));
        // Close this widget before the OpenGL context is destroyed to perform
        // proper cleanup of resources
         connect(selection_controller_->render_view(), SIGNAL(destroying()), SLOT(close()));

        // Add the action as a combo box item in selection controller
        selection_controller_->addComboBoxAction( ui->actionPeakSelection ) ;
    }


    void PeakController::
            mousePressEvent ( QMouseEvent * e )
    {
        if( selection_button_ == e->button() )
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

            if (false == Ui::MouseControl::planePos(
                    e->x(), height() - e->y(),
                    selectionStart.time, selectionStart.scale, r.xscale))
            {
                selectionStart.time = 0.f/0.f;
            }

            selection_controller_->render_view()->userinput_update();
        }
    }


    void PeakController::
            mouseReleaseEvent ( QMouseEvent * e )
    {
        if( selection_button_ == e->button() )
        {
            model()->updateFilter();

            selection_controller_->setCurrentSelection( model()->filter );

            selection_controller_->render_view()->userinput_update();
        }
    }


    void PeakController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        if( e->buttons().testFlag(selection_button_) )
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            r.makeCurrent(); // required for Ui::MouseControl::planePos

        //    TaskTimer tt("moving");

            Heightmap::Position p;
            if (Ui::MouseControl::planePos(
                    e->x(), height() - e->y(),
                    p.time, p.scale, r.xscale))
            {
                if (isnan( selectionStart.time )) // TODO test
                {
                    selectionStart = p;
                }

                float rt = p.time - selectionStart.time;
                float rf = p.scale - selectionStart.scale;
                model()->a = Heightmap::Position(
                        selectionStart.time +  .5f*rt,
                        selectionStart.scale + .5f*rf );
                model()->b = Heightmap::Position(
                        model()->a.time +  .5f*sqrtf(2.f)*rt,
                        model()->a.scale + .5f*sqrtf(2.f)*rf );
            }

            selection_controller_->render_view()->userinput_update();
        }
    }


    void PeakController::
            changeEvent ( QEvent * event )
    {
        if (event->type() & QEvent::EnabledChange)
        {
            view_->enabled = isEnabled();
            emit enabledChanged(isEnabled());
        }
    }


    void PeakController::
            enablePeakSelection(bool active)
    {
        selection_controller_->setCurrentTool( this, active );
    }
*/
}} // namespace Tools::Selections
