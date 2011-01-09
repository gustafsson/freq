#include "graphcontroller.h"

// connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateLayerList(Signal::pOperation)));
// connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateOperationsTree(Signal::pOperation)));
//connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
//connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));

// updateOperationsTree( project->worker.source() );

#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/support/operation-composite.h"

namespace Tools
{
    GraphController::
            GraphController( RenderView* render_view )
                :
                render_view_(render_view),
                worker_(&render_view->model->project()->worker)
    {
        setupGui();
    }


    GraphController::
            ~GraphController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void GraphController::
            redraw_operation_tree()
    {
        Ui::SaweMainWindow* main = render_view_->model->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        Signal::pOperation o = worker_->source();
        QStringList names;
        while(o)
        {
            Tools::Support::OperationSubOperations* subop =
                    dynamic_cast<Tools::Support::OperationSubOperations*>(o.get());
            if (subop)
            {
                names.append( subop->name().c_str() );
            }
            else
            {
                names.append( vartype(*o.get()).c_str() );
            }
            o = o->source();
        }
        ui->operationsTree->clear();
        foreach(QString s, names)
        {
            QTreeWidgetItem* itm = new QTreeWidgetItem(ui->operationsTree);
            itm->setText(0, s);
        }
    }


    void GraphController::
            setupGui()
    {
        connect( worker_, SIGNAL(source_changed()), SLOT(redraw_operation_tree()));
        redraw_operation_tree();
    }

}
