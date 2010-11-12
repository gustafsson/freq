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
    PeakController::
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
        mouseMoveEvent( e );
    }


    void PeakController::
            mouseReleaseEvent ( QMouseEvent * e )
    {
        if (e->button()==selection_button_)
        {
            selection_controller_->setCurrentSelection( model()->spline_model.filter );
        }

        selection_controller_->render_view()->userinput_update();
    }


    void PeakController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        Tools::RenderView &r = *selection_controller_->render_view();

        if (e->buttons().testFlag( selection_button_ ))
        {
            r.makeCurrent();

            GLdouble p[2];
            if (Ui::MouseControl::worldPos( e->x(), height() - e->y(), p[0], p[1], r.xscale))
            {
                Heightmap::Reference ref = r.model->renderer->findRefAtCurrentZoomLevel( p[0], p[1] );
                model()->findAddPeak( ref, Heightmap::Position( p[0], p[1]) );
            }
        }

        r.userinput_update();
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

        if (active)
            selection_controller_->setCurrentSelection( model()->spline_model.filter );
    }

}} // namespace Tools::Selections
