#include "rectangleform.h"
#include "ui_rectangleform.h"

#include "rectanglemodel.h"

#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

namespace Tools {
namespace Selections {

RectangleForm::RectangleForm(RectangleModel* model, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RectangleForm),
    model_(model),
    dontupdate_(false)
{
    ui->setupUi(this);

    updateGui();

    connect(ui->spinBoxStartTime, SIGNAL(valueChanged(double)), SLOT(updateSelection()));
    connect(ui->spinBoxStartFrequency, SIGNAL(valueChanged(double)), SLOT(updateSelection()));
    connect(ui->spinBoxStopTime, SIGNAL(valueChanged(double)), SLOT(updateSelection()));
    connect(ui->spinBoxStopFrequency, SIGNAL(valueChanged(double)), SLOT(updateSelection()));
}


RectangleForm::~RectangleForm()
{
    TaskInfo ti("~RectangleForm");
    delete ui;
}


void RectangleForm::
        updateSelection()
{
    if (dontupdate_)
        return;

    float hza = ui->spinBoxStartFrequency->value();
    float hzb = ui->spinBoxStopFrequency->value();
    model_->a.time = ui->spinBoxStartTime->value();
    model_->b.time = ui->spinBoxStopTime->value();
    model_->a.scale = model_->freqAxis().getFrequencyScalar( hza );
    model_->b.scale = model_->freqAxis().getFrequencyScalar( hzb );

    model_->validate();

    updateGui();

    model_->project()->tools().render_view()->userinput_update();
}


void RectangleForm::
        updateGui()
{
    dontupdate_ = true;
    ui->spinBoxStartFrequency->setMaximum( model_->freqAxis().getFrequency( 1.f ) );
    ui->spinBoxStopFrequency->setMaximum( model_->freqAxis().getFrequency( 1.f ) );
    ui->spinBoxStartFrequency->setMinimum( model_->freqAxis().getFrequency( 0.f ) );
    ui->spinBoxStopFrequency->setMinimum( model_->freqAxis().getFrequency( 0.f ) );

    float hza = model_->freqAxis().getFrequency( model_->a.scale );
    float hzb = model_->freqAxis().getFrequency( model_->b.scale );
    ui->spinBoxStartTime->setValue( model_->a.time );
    ui->spinBoxStopTime->setValue( model_->b.time );
    ui->spinBoxStartFrequency->setValue( hza );
    ui->spinBoxStopFrequency->setValue( hzb );

    QDockWidget* toolWindow = model_->project()->mainWindow()->getItems()->toolPropertiesWindow;
    toolWindow->setWidget( this );
    dontupdate_ = false;
}


} // namespace Selections
} // namespace Tools
