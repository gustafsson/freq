#include "reassigncontroller.h"

// Sonic AWE
#include "sawe/project.h"
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

namespace Tools
{

Reassign::
        Reassign( Sawe::Project* project )
            :
            _model(&project->worker)
{
    setupGui(project);
}


void Reassign::
        setupGui(Sawe::Project* project)
{
    Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(ui->actionTonalizeFilter, SIGNAL(toggled(bool)), SLOT(receiveTonalizeFilter(bool)));
    connect(ui->actionReassignFilter, SIGNAL(toggled(bool)), SLOT(receiveReassignFilter(bool)));
}


void Reassign::
        receiveTonalizeFilter(bool)
{
    Signal::pOperation tonalize( new Filters::Tonalize());
    tonalize->source( _model->source() );
    _model->source( tonalize );
}


void Reassign::
        receiveReassignFilter(bool)
{
    Signal::pOperation reassign( new Filters::Reassign());
    reassign->source( _model->source() );
    _model->source( reassign );
}

} // namespace Tools
