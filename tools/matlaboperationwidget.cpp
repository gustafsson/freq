#include "matlaboperationwidget.h"
#include "ui_matlaboperationwidget.h"

#include "sawe/project.h"

#include <QFileDialog>
#include <QMessageBox>

namespace Tools {

MatlabOperationWidget::MatlabOperationWidget(Sawe::Project* project, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MatlabOperationWidget),
    project(project)
{
    ui->setupUi(this);
    ui->samplerateLabel->setText( QString("1 second is %1 samples.").arg(project->head->head_source()->sample_rate()) );
    connect(ui->browseButton, SIGNAL(clicked()), SLOT(browse()));

    target.reset( new Signal::Target( &project->layers, "Matlab target" ));
    target->findHead( project->head->chain() )->head_source( project->head->head_source() );
}


MatlabOperationWidget::~MatlabOperationWidget()
{
    delete ui;
}


std::string MatlabOperationWidget::
        scriptname()
{
    return ui->scriptname->text().toStdString();
}


void MatlabOperationWidget::
        scriptname(std::string v)
{
    ui->scriptname->setText( QString::fromStdString( v ) );
}


int MatlabOperationWidget::
        chunksize()
{
    return ui->chunksize->value();
}


void MatlabOperationWidget::
        chunksize(int v)
{
    ui->chunksize->setValue( v );
}


bool MatlabOperationWidget::
        computeInOrder()
{
    return ui->computeInOrder->isChecked();
}


void MatlabOperationWidget::
        computeInOrder(bool v)
{
    ui->computeInOrder->setChecked( v );
}


int MatlabOperationWidget::
        redundant()
{
    return ui->redundant->value();
}


void MatlabOperationWidget::
        redundant(int v)
{
    ui->redundant->setValue( v );
}


void MatlabOperationWidget::
        browse()
{
    QString qfilename = QFileDialog::getOpenFileName(
            parentWidget(),
            "Open MATLAB/octave script","",
            "MATLAB/octave script files (*.m)");

    if (!qfilename.isEmpty())
        ui->scriptname->setText( qfilename );
}


void MatlabOperationWidget::
        populateTodoList()
{
    if (project->worker.fetch_todo_list().empty())
    {
        if (operation->invalid_samples())
        {
            if (computeInOrder())
            {
                project->worker.center = 0;
                operation->invalidate_samples(operation->invalid_samples());
                project->worker.target( target );
            }
        }
    }
}


} // namespace Tools
