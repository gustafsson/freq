#include "matlaboperationwidget.h"
#include "ui_matlaboperationwidget.h"

#include "adapters/matlaboperation.h"

#include <QFileDialog>
#include <QMessageBox>

namespace Tools {

MatlabOperationWidget::MatlabOperationWidget(unsigned FS, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MatlabOperationWidget)
{
    ui->setupUi(this);
    ui->samplerateLabel->setText( QString("1 second is %1 samples.").arg(FS) );
    connect(ui->browseButton, SIGNAL(clicked()), SLOT(browse()));
}


MatlabOperationWidget::~MatlabOperationWidget()
{
    delete ui;
}


Signal::pOperation MatlabOperationWidget::
        createMatlabOperation()
{
    if (!QFile::exists( ui->scriptname->text() ))
    {
        QMessageBox::warning( parentWidget(), "Opening file", "Cannot open file '" + ui->scriptname->text() + "'!" );
        return Signal::pOperation();
    }

    Adapters::MatlabOperation* m = new Adapters::MatlabOperation(Signal::pOperation(), ui->scriptname->text().toStdString());
    Signal::pOperation r(m);
    return r;
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


} // namespace Tools
