#include "matlaboperationwidget.h"
#include "ui_matlaboperationwidget.h"
#include "support/commandedit.h"
#include "adapters/readmatlabsettings.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QVBoxLayout>
#include <QPlainTextEdit>
#include <QDockWidget>
#include <QTextDocumentFragment>
#include <QSettings>

namespace Tools {

MatlabOperationWidget::MatlabOperationWidget(Adapters::MatlabFunctionSettings* psettings, Sawe::Project* project, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MatlabOperationWidget),
    project(project),
    octaveWindow(0),
    text(0),
    verticalLayout(0),
    edit(0)
{
    ui->setupUi(this);
    ui->pushButtonRestartScript->setVisible(false);
    ui->pushButtonRestoreChanges->setVisible(false);
    ui->pushButtonShowOutput->setVisible(false);

    connect(ui->browseButton, SIGNAL(clicked()), SLOT(browse()));

    announceInvalidSamplesTimer.setSingleShot( true );
    announceInvalidSamplesTimer.setInterval( 20 );
    connect( &announceInvalidSamplesTimer, SIGNAL(timeout()), SLOT(announceInvalidSamples()));

    connect( ui->scriptname, SIGNAL(textChanged(QString)), SLOT(reloadAutoSettings()));
    connect( ui->scriptname, SIGNAL(textChanged(QString)), SLOT(postRestartScript()));
    connect( ui->scriptname, SIGNAL(returnPressed()), SLOT(restartScript()));
    connect( ui->computeInOrder, SIGNAL(toggled(bool)), SLOT(postRestartScript()));
    connect( ui->chunksize, SIGNAL(valueChanged(int)), SLOT(postRestartScript()));
    connect( ui->redundant, SIGNAL(valueChanged(int)), SLOT(postRestartScript()));
    connect( ui->arguments, SIGNAL(textChanged(QString)), SLOT(postRestartScript()));
    connect( ui->chunksize, SIGNAL(valueChanged(int)), SLOT(chunkSizeChanged()));
    connect( ui->pushButtonRestartScript, SIGNAL(clicked()), SLOT(restartScript()) );
    connect( ui->pushButtonRestoreChanges, SIGNAL(clicked()), SLOT(restoreChanges()) );
    ui->settingsBox->setChecked( true );
    connect( ui->settingsBox, SIGNAL(toggled(bool)), SLOT(settingsVisibleToggled(bool)) );

    ui->settingsBox->setChecked( false );

    QSettings settings;
    settings.beginGroup("MatlabOperationWidget");
    ui->scriptname->setText(        settings.value("scriptname").toString() );
    ui->computeInOrder->setChecked( settings.value("computeInOrder" ).toBool());
    ui->chunksize->setValue(        settings.value("chunksize" ).toInt());
    ui->redundant->setValue(        settings.value("redundant" ).toInt());
    settings.endGroup();

    if (psettings)
    {
        ui->labelEmptyForTerminal->setVisible( false );

        ui->scriptname->setText(        psettings->scriptname().c_str() );
        ui->computeInOrder->setChecked( psettings->computeInOrder() );
        ui->chunksize->setValue(        psettings->chunksize() );
        ui->redundant->setValue(        psettings->redundant() );
        ui->arguments->setText(         psettings->arguments().c_str() );
        ui->scriptname->setEnabled( false );
        ui->settingsBox->setChecked( true );
        ui->settingsBox->setCheckable( false );

        if (psettings->chunksize() == -2)
        {
            // source
            ui->computeInOrder->hide();
            ui->chunksize->hide();
            ui->redundant->hide();
            ui->labelChunkSize->hide();
            ui->labelChunkSizeInfo->hide();
            ui->labelInOrderInfo->hide();
            ui->labelRedundantSamples->hide();
            ui->labelRedundantSamplesInfo->hide();
            window()->resize( 452, 170 );
        }
    }
}


MatlabOperationWidget::
        ~MatlabOperationWidget()
{
    TaskInfo ti("~MatlabOperationWidget");

    hideEvent(0);

    if (octaveWindow)
        delete octaveWindow.data();

    text = 0;
    edit = 0;
    if (operation)
    {
        Adapters::MatlabOperation* o = operation;
        operation = 0;
        o->settings( 0 );
    }

    disconnect( this, SLOT(showOutput()));
    disconnect( this, SLOT(finished(int,QProcess::ExitStatus)));

    delete ui;

    ownOperation.reset();
}


std::string MatlabOperationWidget::
        scriptname()
{
    QString display_path = ui->scriptname->text();

#ifdef _WIN32
    display_path.replace("\\", "/");
#endif

    return operation ? prevsettings.scriptname() : display_path.toStdString();
}


void MatlabOperationWidget::
        scriptname(std::string v)
{
    bool restore = ui->pushButtonRestoreChanges->isEnabled();
    QString display_path = QString::fromStdString( v );

#ifdef _WIN32
    display_path.replace("/", "\\");
#endif

    ui->scriptname->setText( display_path );
    prevsettings.scriptname_ = v;
    ui->pushButtonRestoreChanges->setEnabled(restore);
}


std::string MatlabOperationWidget::
        arguments()
{
    return operation ? prevsettings.arguments() : ui->arguments->text().toStdString();
}


void MatlabOperationWidget::
        arguments(std::string v)
{
    bool restore = ui->pushButtonRestoreChanges->isEnabled();
    ui->arguments->setText( QString::fromStdString( v ) );
    prevsettings.arguments_ = v;
    ui->pushButtonRestoreChanges->setEnabled(restore);
}


int MatlabOperationWidget::
        chunksize()
{
    return operation ? prevsettings.chunksize() : ui->chunksize->value();
}


void MatlabOperationWidget::
        chunksize(int v)
{
    bool restore = ui->pushButtonRestoreChanges->isEnabled();
    ui->chunksize->setValue( v );
    prevsettings.chunksize_ = v;
    ui->pushButtonRestoreChanges->setEnabled(restore);
}


bool MatlabOperationWidget::
        computeInOrder()
{
    return operation ? prevsettings.computeInOrder() : ui->computeInOrder->isChecked();
}


void MatlabOperationWidget::
        computeInOrder(bool v)
{
    bool restore = ui->pushButtonRestoreChanges->isEnabled();
    ui->computeInOrder->setChecked( v );
    prevsettings.computeInOrder_ = v;
    ui->pushButtonRestoreChanges->setEnabled(restore);
}


int MatlabOperationWidget::
        redundant()
{
    return operation ? prevsettings.redundant() :  ui->redundant->value();
}


void MatlabOperationWidget::
        redundant(int v)
{
    bool restore = ui->pushButtonRestoreChanges->isEnabled();
    ui->redundant->setValue( v );
    prevsettings.redundant_ = v;
    ui->pushButtonRestoreChanges->setEnabled(restore);
}


class DummySink: public Signal::Sink
{
public:
    DummySink( unsigned C_) :C_(C_) {}
    virtual bool deleteMe() { return false; }
    virtual void invalidate_samples(const Signal::Intervals& I) { invalid_samples_ |= I; }
    virtual Signal::Intervals invalid_samples() { return invalid_samples_; }
    virtual Signal::pBuffer read( const Signal::Interval& I )
    {
        Signal::pBuffer b = Operation::read(I);
        invalid_samples_ -= b->getInterval();
        return b;
    }
    virtual unsigned num_channels() { return C_; }

private:
    Signal::Intervals invalid_samples_;
    unsigned C_;
};


bool MatlabOperationWidget::
        hasValidTarget()
{
    if (!this->operation)
        return false;

    if (matlabTarget)
    {
        bool isreferenced = false;
        foreach (Signal::pChain c, project->layers.layers())
        {
            if (c!=matlabChain)
                isreferenced |= c->isInChain( matlabChain->root_source() );
        }

        // If the matlab operation is only needed by the MatlabOperationWidget it has been removed, delete this
        if (!isreferenced)
        {
            delete this;
            return false;
        }
        return true;
    }

    Signal::pOperation om;
    foreach(Signal::Operation* c, this->operation->outputs())
    {
        BOOST_ASSERT(c->source().get() == this->operation);
        om = c->source();
    }
    if (!om)
        return false;

    matlabChain.reset( new Signal::Chain(om) );
    Signal::pChainHead ch( new Signal::ChainHead(matlabChain));
    matlabTarget.reset( new Signal::Target(&project->layers, "Matlab", false, false));
    matlabTarget->addLayerHead( ch );

    std::vector<Signal::pOperation> sinks;
    DummySink* ssc = new DummySink( om->num_channels() );
    sinks.push_back( Signal::pOperation( ssc ) );
    matlabTarget->post_sink()->sinks( sinks );

    ui->pushButtonRestartScript->setVisible(true);
    ui->pushButtonRestoreChanges->setVisible(true);
    ui->pushButtonShowOutput->setVisible(true);
    ui->pushButtonRestoreChanges->setEnabled(false);
    ui->pushButtonShowOutput->setEnabled(false);
    ui->labelEmptyForTerminal->setVisible(false);

    return true;
}


QDockWidget* MatlabOperationWidget::
        getOctaveWindow()
{
    return octaveWindow;
}


bool MatlabOperationWidget::
        hasProcess()
{
    return !pid.isNull();
}


void MatlabOperationWidget::
        browse()
{
    QString qfilename = QFileDialog::getOpenFileName(
            parentWidget(),
            "Open MATLAB/octave script", ui->scriptname->text(),
            "MATLAB/octave script files (*.m)");

#ifdef _WIN32
    qfilename.replace("/", "\\");
#endif

    if (!qfilename.isEmpty())
        ui->scriptname->setText( qfilename );
}


void MatlabOperationWidget::
        populateTodoList()
{
    if (hasValidTarget())
    {
        Signal::Intervals needupdate = operation->invalid_returns() | operation->invalid_samples();
        Signal::Interval i = needupdate.coveredInterval();
        if (operation->intervalToCompute( i ).count())
        if (pid && pid->state() != QProcess::NotRunning)
        {
            // restart the timer
            if (!announceInvalidSamplesTimer.isActive())
                announceInvalidSamplesTimer.start();

            if (project->worker.todo_list().empty())
            {
                project->worker.center = 0;
                project->worker.target(matlabTarget);
            }
        }
    }
}


void MatlabOperationWidget::
        announceInvalidSamples()
{
    BOOST_ASSERT(operation);

    Signal::Intervals invalid_returns = operation->invalid_returns();
    Signal::Intervals invalid_samples = operation->invalid_samples();
    Signal::Intervals needupdate = invalid_returns | invalid_samples;

    if (operation->dataAvailable())
    {
        // MatlabOperation calls invalidate_samples which will eventually make
        // RenderView start working if the new data was needed
        project->tools().render_view()->userinput_update( false );
    }

    if (!operation->isWaiting())
    {
        TaskInfo("MatlabOperationWidget needupdate %s", needupdate.toString().c_str());
        operation->OperationCache::invalidate_cached_samples( needupdate );
        matlabTarget->post_sink()->invalidate_samples( needupdate );
        project->tools().render_view()->userinput_update( false );
    }

    Signal::Interval i = needupdate.coveredInterval();
    if (operation->intervalToCompute( i ).count())
    {
        // restart the timer
        announceInvalidSamplesTimer.start();
    }
}


void MatlabOperationWidget::
        invalidateAllSamples()
{
    if (operation)
        operation->invalidate_samples( operation->getInterval() );
}


void MatlabOperationWidget::
        restartScript()
{
    if (operation)
    {
        Adapters::MatlabOperation* t = operation;
        operation = 0;
        if (!prevsettings.scriptname_.empty() && scriptname().empty())
            return;
        prevsettings.scriptname_ = scriptname();
        prevsettings.arguments_ = arguments();
        prevsettings.chunksize_ = chunksize();
        prevsettings.computeInOrder_ = computeInOrder();
        prevsettings.redundant_ = redundant();
        operation = t;

        operation->restart();

        if (octaveWindow)
        {
            if (operation->name().empty())
                octaveWindow->setWindowTitle( "Octave window" );
            else
                octaveWindow->setWindowTitle( QFileInfo(operation->name().c_str()).fileName() );
        }

        if (text)
        {
            text->appendPlainText( QString("\nRestaring script '%1'\n")
                                   .arg(scriptname().c_str()));
            text->moveCursor( QTextCursor::End );
        }
    }
}


void MatlabOperationWidget::
        reloadAutoSettings()
{
    Adapters::ReadMatlabSettings::readSettingsAsync( ui->scriptname->text(), this, SLOT(settingsRead(Adapters::DefaultMatlabFunctionSettings)));
}



void MatlabOperationWidget::
        settingsRead( Adapters::DefaultMatlabFunctionSettings settings )
{
    TaskInfo ti("MatlabOperationWidget: settings");
    TaskInfo("arguments = %s", settings.arguments_.c_str());
    TaskInfo("chunksize = %d", settings.chunksize_);
    TaskInfo("computeInOrder = %d", settings.computeInOrder_);
    TaskInfo("redundant = %d", settings.redundant_);
    TaskInfo("scriptname = %s", settings.scriptname_.c_str());

    arguments(settings.arguments());
    chunksize(settings.chunksize());
    computeInOrder(settings.computeInOrder());
    redundant(settings.redundant());
}


void MatlabOperationWidget::
        postRestartScript()
{
    if (operation)
    {
        ui->pushButtonRestoreChanges->setEnabled(true);
    }
}


void MatlabOperationWidget::
        chunkSizeChanged()
{
    if (ui->chunksize->value()<0)
        ui->chunksize->setSingleStep(1);
    else
        ui->chunksize->setSingleStep(1000);
}


void MatlabOperationWidget::
        restoreChanges()
{
    QWidget* currentFocus = focusWidget();
    scriptname      ( prevsettings.scriptname_ );
    arguments       ( prevsettings.arguments_ );
    chunksize       ( prevsettings.chunksize_ );
    computeInOrder  ( prevsettings.computeInOrder_ );
    redundant       ( prevsettings.redundant_ );
    currentFocus->setFocus();

    ui->pushButtonRestoreChanges->setEnabled(false);
}


void MatlabOperationWidget::
        settingsVisibleToggled(bool v)
{
    ui->settingsBox->setMaximumHeight(v ? 524287 : 20);
    int h = ui->labelEmptyForTerminal->isVisible() ? 0 : ui->labelEmptyForTerminal->height();

    // xkcd.com/974/
    if (v)
        this->window()->resize( 452, 478-h );
    else
        this->window()->resize( 452, 170-h );
}


void MatlabOperationWidget::
        setProcess(QProcess* pid)
{
    prevsettings.pid_ = pid;
    this->pid = pid;
    connect( pid, SIGNAL(readyRead()), SLOT(showOutput()));
    connect( pid, SIGNAL(finished( int , QProcess::ExitStatus )), SLOT(finished(int,QProcess::ExitStatus)));

    {
        Adapters::MatlabOperation* t = operation;
        operation = 0;
        if (!prevsettings.scriptname_.empty() && scriptname().empty())
            return;
        prevsettings.scriptname_ = scriptname();
        prevsettings.arguments_ = arguments();
        prevsettings.chunksize_ = chunksize();
        prevsettings.computeInOrder_ = computeInOrder();
        prevsettings.redundant_ = redundant();
        operation = t;
    }

    ui->pushButtonRestoreChanges->setEnabled(false);
}


void MatlabOperationWidget::
        finished ( int exitCode, QProcess::ExitStatus /*exitStatus*/ )
{
    if (!octaveWindow)
        return;

    if (text)
    {
        text->appendPlainText( QString("\nThe process ended %1with exit code %2")
                               .arg(0 != exitCode ? "unexpectedly " : "" )
                               .arg(exitCode));
        text->moveCursor( QTextCursor::End );

    }

    if (edit)
        edit->setEnabled( false );
}


void MatlabOperationWidget::
        checkOctaveVisibility()
{
    if (octaveWindow)
        ui->pushButtonShowOutput->setChecked( octaveWindow->isVisible() );
}


void MatlabOperationWidget::
        hideEvent ( QHideEvent * /*event*/ )
{
    QSettings settings;
    // this->saveGeometry() doesn't save child widget states
    settings.beginGroup("MatlabOperationWidget");
    settings.setValue("scriptname", ui->scriptname->text() );
    settings.setValue("computeInOrder", ui->computeInOrder->isChecked() );
    settings.setValue("chunksize", ui->chunksize->value() );
    settings.setValue("redundant", ui->redundant->value() );
    settings.endGroup();
}


void MatlabOperationWidget::
        showOutput()
{
    if (0 == octaveWindow && text != 0)
        return;

    if (0 == operation)
        return;

    if (0==octaveWindow)
    {
        octaveWindow = new QDockWidget(project->mainWindow());
        octaveWindow->setObjectName(QString::fromUtf8("octaveWindow"));
        octaveWindow->setMinimumSize(QSize(113, 113));
        octaveWindow->setFeatures(QDockWidget::AllDockWidgetFeatures);
        octaveWindow->setAllowedAreas(Qt::AllDockWidgetAreas);        
        if (operation->name().empty())
            octaveWindow->setWindowTitle( "Octave window" );
        else
            octaveWindow->setWindowTitle( QFileInfo(operation->name().c_str()).fileName() );
        QWidget* dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout = new QVBoxLayout(dockWidgetContents);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        text = new QPlainTextEdit;
        text->setReadOnly( true );
        verticalLayout->addWidget( text );

        if (operation->name().empty())
        {
            edit = new Support::CommandEdit;
            verticalLayout->addWidget( edit );
            connect( edit, SIGNAL(returnPressed()), SLOT( sendCommand() ));
            edit->setText( "Enter commands here" );
            edit->setFocus();
            edit->setSelection(0, edit->text().size());
        }

        octaveWindow->setWidget(dockWidgetContents);
        project->mainWindow()->addDockWidget( Qt::BottomDockWidgetArea, octaveWindow );
        octaveWindow->hide();

        connect( ui->pushButtonShowOutput, SIGNAL(toggled(bool)), octaveWindow.data(), SLOT(setVisible(bool)));
        connect( octaveWindow.data(), SIGNAL(visibilityChanged(bool)), SLOT(checkOctaveVisibility()));
        ui->pushButtonShowOutput->setEnabled( true );
    }

    octaveWindow->show();

    QByteArray ba = pid->readAllStandardOutput();
    QString s( ba );
    text->moveCursor( QTextCursor::End );
    text->insertPlainText( s );
    text->moveCursor( QTextCursor::End );
    TaskInfo("Matlab output (%p): %s", this, s.replace("\r","").toStdString().c_str());
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
