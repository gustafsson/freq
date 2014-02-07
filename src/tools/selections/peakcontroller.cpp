#if 0
#include "peakcontroller.h"
#include "peakmodel.h"

// Sonic AWE tools
#include "tools/selectioncontroller.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/renderview.h"

// Sonic AWE
#include "heightmap/renderer.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/collection.h"

// gpumisc
#include "TaskTimer.h"

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
        TaskInfo(__FUNCTION__);
    }


    void PeakController::
            setupGui()
    {
        Ui::SaweMainWindow* main = selection_controller_->model()->project()->mainWindow();
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
            selection_controller_->setCurrentSelection( model()->updateFilter() );
        }

        selection_controller_->render_view()->redraw();
    }


    void PeakController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        Tools::RenderView &r = *selection_controller_->render_view();

        if (e->buttons().testFlag( selection_button_ ))
        {
            Heightmap::Collection::Ptr c = r.model->collections()[0];

            Heightmap::Position p = r.getHeightmapPos( e->localPos () );
            Heightmap::Reference ref = r.findRefAtCurrentZoomLevel( p );
            if([&](){
                Heightmap::Collection::ReadPtr rc (c);
                Heightmap::ReferenceInfo ri(ref, rc->block_layout (), rc->visualization_params());
                return ri.containsPoint(p);
            }())
            {
                model()->findAddPeak( c, ref, p );
            }
        }

        r.redraw();
    }


    void PeakController::
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


    void PeakController::
            enablePeakSelection(bool active)
    {
        if (active)
        {
            selection_controller_->setCurrentSelection( model()->updateFilter() );
            selection_controller_->setCurrentTool( this, active );
        }
    }

}} // namespace Tools::Selections
#endif
