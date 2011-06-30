#include "selectionviewinfo.h"
#include "ui_selectionviewinfo.h"

#include "selectionmodel.h"

#include "sawe/project.h"
#include "support/computerms.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "signal/target.h"
#include "signal/sinksourcechannels.h"

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
    connect(this, SIGNAL(visibilityChanged(bool)), SLOT(setVisible(bool)));
    ui_->actionSelection_Info->setChecked( false );

    project->mainWindow()->getItems()->menu_Windows->addAction(ui_->actionSelection_Info);
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

    if (!model_->current_selection() || !isVisibleTo(parentWidget()))
    {
        target_.reset();
        return;
    }

    Signal::pOperation selection = model_->current_selection_copy();
    target_.reset( new Target(&model_->project()->layers, "SelectionViewInfo", false) );
    target_->addLayerHead(project_->head);

    std::vector<Signal::pOperation> svso;
    // A target must have a post sink with a sink that can tell what to work on.
    // A target is defined by the sinks that request data to be computed.
    Signal::pOperation infoOperation(new SelectionViewInfoSink(this));
    svso.push_back( infoOperation );
    target_->post_sink()->sinks( svso );
    target_->post_sink()->filter(selection);
    target_->post_sink()->invalidate_samples(~selection->zeroed_samples_recursive());

    project_->targets.insert( target_ );
}


void SelectionViewInfo::
        setVisible(bool visible)
{
    selectionChanged();
    QDockWidget::setVisible(visible);
}


SelectionViewInfoSink::
        SelectionViewInfoSink( SelectionViewInfo* info )
            :
            info_(info)
{
    rms_.reset(new Support::ComputeRms(pOperation()));
    Operation::source(rms_);
}


pBuffer SelectionViewInfoSink::
        read( const Interval& I )
{
    pBuffer b = Operation::read(I);
    Support::ComputeRms* rms = dynamic_cast<Support::ComputeRms*>(rms_.get());
    Intervals all = this->getInterval() - this->zeroed_samples_recursive();
    Intervals not_processed = all-rms->rms_I;
    double P0 = 1;
    double P = rms->rms;
    double db = 10*log10(P/P0);
    QString text;
    text += QString("Mean intensity: %1 db").arg(db);
    if (not_processed)
        text += QString(" (%1%)")
                       .arg(1.f - not_processed.count()/(float)all.count());
    text +="\n";
    text += QString("Selection length: %1").arg(QString::fromStdString(SourceBase::lengthLongFormat(all.count()/sample_rate())));
    text += "\n";
    text += QString("Total signal length: %1").arg(QString::fromStdString(lengthLongFormat()));

    info_->setText(text);

    missing_ -= b->getInterval();
    return b;
}


void SelectionViewInfoSink::
        source(pOperation v)
{
    rms_->source(v);
}


void SelectionViewInfoSink::
        invalidate_samples(const Intervals& I)
{
    missing_ |= I;
}


Intervals SelectionViewInfoSink::
        invalid_samples()
{
    return missing_;
}

} // namespace Tools
