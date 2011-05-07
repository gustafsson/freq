#include "selectionviewinfo.h"
#include "ui_selectionviewinfo.h"

#include "selectionmodel.h"

#include "sawe/project.h"
#include "support/computerms.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "signal/target.h"

using namespace Signal;

namespace Tools {

SelectionViewInfo::
        SelectionViewInfo(Sawe::Project* project, SelectionModel* model)
            :
    QDockWidget(project->mainWindow()),
    project_(project),
    model_(model),
    ui_(new Ui::SelectionViewInfo)
{
    ui_->setupUi(this);

    connect(model, SIGNAL(selectionChanged()), SLOT(selectionChanged()));

    selectionChanged();

    hide();
    project->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, this);

    connect(ui_->actionSelection_Info, SIGNAL(toggled(bool)), this, SLOT(setVisible(bool)));
    connect(ui_->actionSelection_Info, SIGNAL(triggered()), this, SLOT(raise()));
    connect(this, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibility(bool)));
    ui_->actionSelection_Info->setChecked( false );

    project->mainWindow()->getItems()->menuWindows->addAction(ui_->actionSelection_Info);
}


SelectionViewInfo::
        ~SelectionViewInfo()
{
    delete ui_;
}


void SelectionViewInfo::
        setText(QString text)
{
    ui_->textEdit->setText( text );
}


void SelectionViewInfo::
        selectionChanged()
{
    if (target_)
        project_->targets.erase( target_ );

    if (!model_->current_selection())
    {
        target_.reset();
        return;
    }

    target_.reset( new OperationTarget(model_->current_selection_copy(), "SelectionViewInfo") );
    project_->targets.insert( target_ );
}


SelectionViewInfoOperation::
        SelectionViewInfoOperation( pOperation o, SelectionViewInfo* info )
            :
            Operation(pOperation()),
            info_(info)
{
    rms_.reset(new Support::ComputeRms(o));
    source(rms_);
}


pBuffer SelectionViewInfoOperation::
        read( const Interval& I )
{
    pBuffer b = Operation::read(I);
    Support::ComputeRms* rms = dynamic_cast<Support::ComputeRms*>(rms_.get());
    Intervals all = this->getInterval() - this->zeroed_samples_recursive();
    Intervals not_processed = all-rms->rms_I;
    QString text;
    text += QString("RMS %1").arg(rms->rms);
    if (!not_processed.empty())
        text += QString(" (%.1f2%%)").arg(rms->rms)
                       .arg(1.f - not_processed.count()/(float)all.count());

    text += "\n";
    text += QString("Selection length: %1").arg(QString::fromStdString(SourceBase::lengthLongFormat(all.count()/sample_rate())));
    text += "\n";
    text += QString("Total signal length: %1").arg(QString::fromStdString(lengthLongFormat()));

    info_->setText(text);

    return b;
}

} // namespace Tools
