#include "matlabcontroller.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

namespace Tools {

MatlabController::
        MatlabController( Sawe::Project* project )
            :
            _model(&project->worker)
{
    setupGui(project);
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
        ((Adapters::MatlabOperation*)_matlaboperation.get())->restart();
        return;
    }

    _matlaboperation.reset( new Adapters::MatlabOperation( _model->source(), "matlaboperation") );
    _model->source( _matlaboperation );

    // Render view will be updated by invalidating some parts in sinks of worker
    _model->postSink()->invalidate_samples(_matlaboperation->affected_samples());
}


void MatlabController::
        receiveMatlabFilter(bool)
{
    if (_matlabfilter)
    {
        // Already created, make it re-read the script
        dynamic_cast<Adapters::MatlabFilter*>(_matlabfilter.get())->restart();
        return;
    }

    switch(1) {
    case 1: // Everywhere
        {
            _matlabfilter.reset( new Adapters::MatlabFilter( "matlabfilter" ) );
            _matlabfilter->source( _model->source() );
            _model->source( _matlabfilter );

            // Make sure the worker runs in a separate thread
            _model->start();
        break;
        }
/*    case 2: // Only inside selection
        {
            Signal::pOperation s( new Adapters::MatlabFilter( "matlabfilter" ));
            _matlabfilter->source( _model->source() );

            // TODO Fetch selection
            Signal::PostSink* postsink = project->tools().selection_model.getPostSink();

            Filters::EllipseFilter* e = dynamic_cast<Filters::EllipseFilter*>(postsink->filter().get());
            if (e)
                e->_save_inside = true;

            _matlabfilter = postsink->filter();
            postsink->filter(Signal::pOperation());
            _matlabfilter->source( s );

            b->source( _matlabfilter );
            _model->source( _matlabfilter );
            break;
        }*/
    }

    // Render view will be updated by invalidating some parts in sinks of worker
    _model->postSink()->invalidate_samples(_matlabfilter->affected_samples());
}


} // namespace Tools
