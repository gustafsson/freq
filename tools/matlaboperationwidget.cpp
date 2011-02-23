#include "matlaboperationwidget.h"
#include "ui_matlaboperationwidget.h"
#include "support/commandedit.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QVBoxLayout>
#include <QPlainTextEdit>
#include <QDockWidget>
#include <QTextDocumentFragment>

namespace Tools {

MatlabOperationWidget::MatlabOperationWidget(Sawe::Project* project, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MatlabOperationWidget),
    project(project),
    octaveWindow(0),
    text(0),
    edit(0)
{
    ui->setupUi(this);
    ui->samplerateLabel->setText( QString("1 second is %1 samples.").arg(project->head->head_source()->sample_rate()) );
    connect(ui->browseButton, SIGNAL(clicked()), SLOT(browse()));

    //target.reset( new Signal::Target( &project->layers, "Matlab target" ));
    //target->findHead( project->head->chain() )->head_source( project->head->head_source() );

    announceInvalidSamplesTimer.setSingleShot( true );
    announceInvalidSamplesTimer.setInterval( 200 );
    connect( &announceInvalidSamplesTimer, SIGNAL(timeout()), SLOT(announceInvalidSamples()));
}


MatlabOperationWidget::
        ~MatlabOperationWidget()
{
    TaskInfo ti("~MatlabOperationWidget");
    TaskInfo(".");
    octaveWindow = 0;
    text = 0;
    edit = 0;
    operation = 0;

    disconnect( this, SLOT(showOutput()));
    disconnect( this, SLOT(finished(int,QProcess::ExitStatus)));

    delete ui;

    ownOperation.reset();
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
        if (operation && operation->invalid_returns() && pid->state() != QProcess::NotRunning)
        {
            // restart the timer
            announceInvalidSamplesTimer.start();
        }
    }
}


void MatlabOperationWidget::
        announceInvalidSamples()
{
    if (operation->invalid_returns())
        operation->invalidate_samples( operation->invalid_returns() );
}


void MatlabOperationWidget::
        setProcess(QProcess* pid)
{
    this->pid = pid;
    connect( pid, SIGNAL(readyRead()), SLOT(showOutput()));
    connect( pid, SIGNAL(finished( int , QProcess::ExitStatus )), SLOT(finished(int,QProcess::ExitStatus)));
}


void MatlabOperationWidget::
        finished ( int exitCode, QProcess::ExitStatus exitStatus )
{
    if (!octaveWindow)
        return;

    if (text)
    {
        text->appendPlainText( QString("\nThe process ended with exit code %1:%2")
                               .arg(exitCode)
                               .arg(QProcess::NormalExit == exitStatus ? 0 : 1 ));
        text->moveCursor( QTextCursor::End );
    }

    if (edit)
        edit->setEnabled( false );
}


void MatlabOperationWidget::
        showOutput()
{
    if (0 == octaveWindow && text != 0)
        return;

    if (0==octaveWindow)
    {
        octaveWindow = new QDockWidget(project->mainWindow());
        octaveWindow->setObjectName(QString::fromUtf8("octaveWindow"));
        octaveWindow->setMinimumSize(QSize(113, 113));
        octaveWindow->setFeatures(QDockWidget::AllDockWidgetFeatures);
        octaveWindow->setAllowedAreas(Qt::AllDockWidgetAreas);
        if (scriptname().empty())
            octaveWindow->setWindowTitle( "Octave window" );
        else
            octaveWindow->setWindowTitle( QFileInfo(scriptname().c_str()).fileName() );
        QWidget* dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        QVBoxLayout* verticalLayout = new QVBoxLayout(dockWidgetContents);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        text = new QPlainTextEdit;
        text->setReadOnly( true );
        verticalLayout->addWidget( text );

        if (scriptname().empty())
        {
            edit = new Support::CommandEdit;
            verticalLayout->addWidget( edit );
            connect( edit, SIGNAL(returnPressed()), SLOT( sendCommand() ));
        }

        octaveWindow->setWidget(dockWidgetContents);
        project->mainWindow()->addDockWidget( Qt::BottomDockWidgetArea, octaveWindow );
        octaveWindow->hide();
    }
    octaveWindow->show();

    QByteArray ba = pid->readAllStandardOutput();
    QString s( ba );
    text->moveCursor( QTextCursor::End );
    text->insertPlainText( s );
    text->moveCursor( QTextCursor::End );
    TaskInfo("Matlab output (%p): %s", this, s.toStdString().c_str());
}


void MatlabOperationWidget::
        sendCommand()
{
    QString command = edit->text() + "\n";
    pid->write( command.toLocal8Bit() );
    text->moveCursor( QTextCursor::End );
    text->insertPlainText( command );
    text->moveCursor( QTextCursor::End );
    TaskInfo("Matlab command (%p): %s", this, edit->text().toStdString().c_str());
    edit->clear();
}

} // namespace Tools
