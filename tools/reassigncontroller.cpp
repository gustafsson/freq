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
            _project_chainhead(project->head)
{
    setupGui(project);
}


void Reassign::
        setupGui(Sawe::Project* project)
{
    Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(ui->actionTonalizeFilter, SIGNAL(triggered()), SLOT(receiveTonalizeFilter()));
    connect(ui->actionReassignFilter, SIGNAL(triggered()), SLOT(receiveReassignFilter()));
}


void Reassign::
        receiveTonalizeFilter()
{
    Signal::pOperation tonalize( new Filters::Tonalize());
    _project_chainhead->appendOperation( tonalize );
}


void Reassign::
        receiveReassignFilter()
{
    Signal::pOperation reassign( new Filters::Reassign());
    _project_chainhead->appendOperation( reassign );
}

} // namespace Tools
