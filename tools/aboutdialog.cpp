#include "aboutdialog.h"
#include "ui_aboutdialog.h"
#include "ui_mainwindow.h"
#include "sawe/application.h"

namespace Tools
{

AboutDialog::AboutDialog(Sawe::Project* project) :
    QDialog(project->mainWindow()),
    ui(new Ui::AboutDialog)
{
    ui->setupUi(this);

    ui->labelVersion->setText( QString::fromStdString( Sawe::Application::version_string() ) );
    ui->labelTimestamp->setText( QString("Built on %1 at %2 from revision %3").arg(__DATE__).arg(__TIME__).arg(SONICAWE_REVISION) );

    Ui::MainWindow* main_ui = project->mainWindow()->getItems();
    connect(main_ui->actionAbout, SIGNAL(triggered()), SLOT(show()));
    connect(ui->buttonBox, SIGNAL(clicked()), SLOT(close()));
}

AboutDialog::~AboutDialog()
{
    delete ui;
}

} // namespace Tools
