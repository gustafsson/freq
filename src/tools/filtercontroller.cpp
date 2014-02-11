#include "filtercontroller.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "sawe/configuration.h"

#include "filters/absolutevalue.h"
#include "filters/envelope.h"
#include "tfr/transformoperation.h"

namespace Tools {

FilterController::
        FilterController(Sawe::Project* p)
    :
      project_(p)
{
    Ui::MainWindow* items = p->mainWindow ()->getItems ();
    QMenu* filterMenu = items->menuTools->findChild<QMenu*>("filtermenu");
    if (!filterMenu)
        filterMenu = items->menuTools->addMenu ("&Filters");
    connect(filterMenu->addAction ("|y|"), SIGNAL(triggered()), SLOT(addAbsolute()));
    if (!Sawe::Configuration::feature("stable")) {
        connect(filterMenu->addAction ("envelope"), SIGNAL(triggered()), SLOT(addEnvelope()));
    }
}


void FilterController::
        addAbsolute()
{
    project_->appendOperation ( Signal::OperationDesc::Ptr(new Filters::AbsoluteValueDesc() ));
}


void FilterController::
        addEnvelope()
{
    Tfr::ChunkFilterDesc::Ptr cfd(new Filters::EnvelopeDesc());
    Signal::OperationDesc::Ptr o(new Tfr::TransformOperationDesc(cfd));
    project_->appendOperation ( o );
}


} // namespace Tools
