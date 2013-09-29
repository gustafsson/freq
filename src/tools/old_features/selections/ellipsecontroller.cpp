#if 0
#include "ellipsecontroller.h"
#include "ellipsemodel.h"

// Sonic AWE
#include "tools/selectioncontroller.h"
#include "filters/ellipse.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/renderview.h"

// gpumisc
#include "TaskTimer.h"

// Qt
#include <QMouseEvent>

using namespace Tools;

namespace Tools { namespace Selections
{
    EllipseController::
            EllipseController(
                    EllipseView* view,
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


    EllipseController::
            ~EllipseController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void EllipseController::
            setupGui()
    {
        Ui::SaweMainWindow* main = selection_controller_->model()->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        // Connect enabled/disable actions,
        // 'enableEllipseSelection' sets/unsets this as current tool when
        // the action is checked/unchecked.
        connect(ui->actionEllipseSelection, SIGNAL(toggled(bool)), SLOT(enableEllipseSelection(bool)));
        connect(this, SIGNAL(enabledChanged(bool)), ui->actionEllipseSelection, SLOT(setChecked(bool)));

        // Paint the ellipse when render view paints
        connect(selection_controller_->render_view(), SIGNAL(painting()), view_, SLOT(draw()));
        // Close this widget before the OpenGL context is destroyed to perform
        // proper cleanup of resources
        connect(selection_controller_->render_view(), SIGNAL(destroying()), SLOT(close()));


        connect(selection_controller_->model(), SIGNAL(selectionChanged()), SLOT(selectionChanged()));

        // Add the action as a combo box item in selection controller
        selection_controller_->addComboBoxAction( ui->actionEllipseSelection ) ;
    }


    void EllipseController::
            mousePressEvent ( QMouseEvent * e )
    {
        if (e->button() == selection_button_)
        {
            Tools::RenderView &r = *selection_controller_->render_view();
            bool success;
            selectionStart = r.getPlanePos( e->posF(), &success);
            if (!success)
            {
                selectionStart.time = -FLT_MAX;
            }
        }

        selection_controller_->render_view()->userinput_update();
    }


    void EllipseController::
            mouseReleaseEvent ( QMouseEvent * e )
    {
        if (e->button() == selection_button_)
        {
            selection_controller_->setCurrentSelection( model()->updateFilter() );
        }

        selection_controller_->render_view()->userinput_update();
    }


    void EllipseController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        if (e->buttons().testFlag( selection_button_ ))
        {
            Tools::RenderView &r = *selection_controller_->render_view();

        //    TaskTimer tt("moving");
            bool success;
            Heightmap::Position p = r.getPlanePos( e->posF(), &success);

            if (success)
            {
                if (selectionStart.time == -FLT_MAX) // TODO test
                {
                    selectionStart = p;
                }

                float rt = p.time - selectionStart.time;
                float rf = p.scale - selectionStart.scale;
                model()->centre = Heightmap::Position(
                        selectionStart.time +  .5f*rt,
                        selectionStart.scale + .5f*rf );
                model()->centrePlusRadius = Heightmap::Position(
                        model()->centre.time +  .5f*sqrtf(2.f)*rt,
                        model()->centre.scale + .5f*sqrtf(2.f)*rf );
            }
        }

        selection_controller_->render_view()->userinput_update();
    }


    void EllipseController::
            changeEvent ( QEvent * event )
    {
        if (event->type() == QEvent::ParentChange)
        {
            view_->visible = 0!=parent();
        }

        if (event->type() == QEvent::EnabledChange)
        {
            view_->enabled = isEnabled();
            emit enabledChanged(isEnabled());
        }
    }


    void EllipseController::
            enableEllipseSelection(bool active)
    {
        if (active)
        {
            selection_controller_->setCurrentTool( this, active );
            selection_controller_->setCurrentSelection( model()->updateFilter() );
        }
    }


    void EllipseController::
            selectionChanged()
    {
        Signal::pOperation o = selection_controller_->model()->current_selection();
        model()->tryFilter( o );
    }

}} // namespace Tools::Selections
#endif
