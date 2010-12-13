#include "matlabcontroller.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

#include "heightmap/collection.h"
#include "tfr/cwt.h"

namespace Tools {

MatlabController::
        MatlabController( Sawe::Project* project, RenderView* render_view )
            :
            worker_(&project->worker),
            render_view_(render_view)
{
    setupGui(project);
}


MatlabController::
        ~MatlabController()
{
    TaskInfo("~MatlabController");
}


void MatlabController::
        setupGui(Sawe::Project* project)
{
    Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(ui->actionMatlabOperation, SIGNAL(triggered(bool)), SLOT(receiveMatlabOperation(bool)));
    connect(ui->actionMatlabFilter, SIGNAL(triggered(bool)), SLOT(receiveMatlabFilter(bool)));
}



void MatlabController::
        receiveMatlabOperation(bool)
{
    if (_matlaboperation)
    {
        // Already created, make it re-read the script
        dynamic_cast<Adapters::MatlabOperation*>(_matlaboperation.get())->restart();
        worker_->invalidate_post_sink(_matlaboperation->affected_samples());
    }
    else
    {
        _matlaboperation.reset( new Adapters::MatlabOperation( Signal::pOperation(), "matlaboperation") );
        worker_->appendOperation( _matlaboperation );
    }

    render_view_->userinput_update();
}


void MatlabController::
        receiveMatlabFilter(bool)
{
    if (_matlabfilter)
    {
        // Already created, make it re-read the script
        dynamic_cast<Adapters::MatlabFilter*>(_matlabfilter.get())->restart();
        worker_->invalidate_post_sink(_matlabfilter->affected_samples());
    }
    else
    {
        _matlabfilter.reset( new Adapters::MatlabFilter( "matlabfilter" ) );
        worker_->appendOperation( _matlabfilter );

        // Make sure the worker runs in a separate thread
        Tfr::Cwt::Singleton().gc();
        worker_->start();
    }

    render_view_->userinput_update();}


} // namespace Tools
