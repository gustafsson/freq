#include "matlabcontroller.h"
#include "matlaboperationwidget.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "adapters/readmatlabsettings.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "support/operation-composite.h"
#include "tools/support/plotlines.h"

#include "heightmap/collection.h"
#include "tfr/cwt.h"
#include "sawe/application.h"
#include "signal/operation-basic.h"

// boost
#include <boost/foreach.hpp>

// Qt
#include <QDialogButtonBox>
#include <QFile>
#include <QSharedPointer>
#include <QDir>
#include <QRegExp>
#include <QDateTime>
#include <QSettings>
#include <QErrorMessage>
#include <QToolBar>

namespace Tools {


MatlabController::
        MatlabController( Sawe::Project* project, RenderView* render_view )
            :
            project_(project),
            render_view_(render_view)
{
    setupGui();

    // Clean up old h5 files that were probably left from a previous crash

    QDateTime now = QDateTime::currentDateTime();
    foreach (QFileInfo s, QDir::current().entryInfoList( QStringList("*.h5") ))
    {
        if (QRegExp(".*\\.0x[0-9a-f]{6,16}\\.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.0x[0-9a-f]{6,16}\\.h5.result.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.[0-9A-F]{8}\\.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.[0-9A-F]{8}\\.h5.result.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.[0-9A-F]{16}\\.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.[0-9A-F]{16}\\.h5.result.h5").exactMatch(s.fileName()))
        {
            // Delete it only if it was created more than 15 minutes ago,
            // because other instances of Sonic AWE might still be running.
            QDateTime created = s.created();
            int diff = created.secsTo(now);
            if (15*60 < diff)
            {
                TaskInfo("Removing %s", s.filePath().toStdString().c_str());
                ::remove( s.fileName().toStdString().c_str() );
            }
        }
    }


    // create gui for operations already loaded
    createView();
}


MatlabController::
        ~MatlabController()
{
    TaskInfo("~MatlabController");
}


void MatlabController::
        setupGui()
{
    ::Ui::SaweMainWindow* main = project_->mainWindow();
    ::Ui::MainWindow* ui = main->getItems();

    //connect(ui->actionToogleMatlabToolBox, SIGNAL(toggled(bool)), ui->toolBarMatlab, SLOT(setVisible(bool)));
    //connect(ui->toolBarMatlab, SIGNAL(visibleChanged(bool)), ui->actionToogleMatlabToolBox, SLOT(setChecked(bool)));

    connect(ui->actionMatlabOperation, SIGNAL(triggered()), SLOT(receiveMatlabOperation()));
    connect(ui->actionMatlabFilter, SIGNAL(triggered()), SLOT(receiveMatlabFilter()));

    std::set<Signal::pChain> ch = project_->layers.layers();
    for (std::set<Signal::pChain>::iterator itr = ch.begin(); itr != ch.end(); ++itr)
    {
        for (Signal::pOperation o = (*itr)->tip_source(); o; o=o->source() )
        {
            if (Adapters::MatlabOperation* m = dynamic_cast<Adapters::MatlabOperation*>( o.get()))
            {
                prepareLogView( m );
            }
        }
    }

    connect( project_->head.get(), SIGNAL(headChanged()), SLOT(tryHeadAsMatlabOperation()));


    updateScriptsMenu();
}


void MatlabController::
        prepareLogView( Adapters::MatlabOperation*m )
{
    MatlabOperationWidget* settings = new MatlabOperationWidget( m->settings(), project_ );
    settings->operation = m;
    m->settings( settings );

    connect( render_view_, SIGNAL(populateTodoList()), settings, SLOT(populateTodoList()));
}


void MatlabController::
        updateScriptsMenu()
{
    ::Ui::SaweMainWindow* main = project_->mainWindow();
    ::Ui::MainWindow* ui = main->getItems();

    if (!scripts_)
    {
        scripts_ = new QMenu( "Matlab/Octave &scripts",  ui->menuTools );
        ui->menuTools->insertMenu( ui->menuToolbars->menuAction(), scripts_ );
    }
    scripts_->clear();
    scripts_->insertAction( 0, ui->actionMatlabOperation );

    int i = 0;

    QSettings state;
    state.beginGroup("MatlabOperation");
    if (!state.childGroups().empty())
    {
        scripts_->addSeparator();

        QStringList G = state.childGroups();
        G.sort();
        foreach (QString g, G)
        {
            state.beginGroup(g);
            QString path = state.value("path").toString();
            state.endGroup();
            if (!QFile::exists(path))
            {
                if (QMessageBox::Yes == QMessageBox::question( main, "Can't find script", QString("Couldn't find script \"%1\" at \"%2\". Do you want to remove this item from the menu?").arg(g).arg(path), QMessageBox::Yes, QMessageBox::No ))
                {
                    state.remove(g);
                    continue;
                }
            }

            i++;
            QAction* action = new QAction(QString("%1%2. %3").arg(i<10?"&":"").arg(i).arg(g), scripts_ );
            action->setData(g);
            scripts_->addAction( action );
            connect( action, SIGNAL(triggered()), SLOT(createFromAction()));
        }
    }
    state.endGroup();


    QString scriptDirs[] =
    {
        "/usr/share/sonicawe/scripts",
        Sawe::Application::log_directory() + QDir::separator() + "scripts",
        QDir::currentPath() + QDir::separator() + "scripts"
    };


    QFileInfoList scriptfilesinfo;
    BOOST_FOREACH(QString& dir, scriptDirs)
        if (QDir(dir).exists())
            scriptfilesinfo.append(QDir(dir, "*.m").entryInfoList());

    QStringList scriptfiles;
    foreach(QFileInfo info, scriptfilesinfo)
        scriptfiles.append( info.absoluteFilePath() );

    std::sort( scriptfiles.begin(), scriptfiles.end() );

    TaskInfo ti("Found %d candidates for plugins", scriptfiles.size());

    if (!scriptfiles.empty())
        scripts_->addSeparator();

    foreach(QString info, scriptfiles)
    {
        TaskInfo("%s", info.toLatin1().data());
        Adapters::ReadMatlabSettings::readSettingsAsync(info, this, SLOT(settingsRead(Adapters::DefaultMatlabFunctionSettings)));
    }
}


void MatlabController::
        createFromAction()
{
    QAction* a = dynamic_cast<QAction*>(sender());
    BOOST_ASSERT( a );

    QSettings state;
    state.beginGroup("MatlabOperation");
    state.beginGroup( a->data().toString() );

    TaskInfo ti("createFromAction %s", a->data().toString().toStdString().c_str() );

    Adapters::DefaultMatlabFunctionSettings settings;
    settings.arguments( state.value("arguments").toString().toStdString() );
    settings.chunksize( state.value("chunksize").toInt() );
    settings.computeInOrder( state.value("computeInOrder").toBool() );
    settings.operation = 0;
    settings.redundant( state.value("redundant").toInt() );
    settings.scriptname( state.value("path").toString().toStdString() );

    createFromSettings( settings );
}


void MatlabController::
        settingsRead( Adapters::DefaultMatlabFunctionSettings settings )
{
    TaskInfo ti("settingsRead %s", settings.scriptname().c_str() );

    Adapters::ReadMatlabSettings* read = dynamic_cast<Adapters::ReadMatlabSettings*>(sender());
    BOOST_ASSERT( read );

    QFileInfo info(settings.scriptname().c_str());

    QAction* action = 0;

    unsigned i = scripts_->actions().count()-2; // 2 separators and 1 action to create new operations, makes 'i' the next script number
    QString actionText = QString("%1%2. %3").arg(i<10?"&":"").arg(i).arg(info.baseName());

    if (!read->iconpath().empty())
    {
        if (!scriptsToolbar_)
        {
            ::Ui::SaweMainWindow* main = project_->mainWindow();
            scriptsToolbar_ = new Support::ToolBar(main);
            scriptsToolbar_->setObjectName(QString::fromUtf8("scriptsToolbar"));
            scriptsToolbar_->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0, QApplication::UnicodeUTF8));
            scriptsToolbar_->setEnabled(true);
            scriptsToolbar_->setContextMenuPolicy(Qt::NoContextMenu);
            scriptsToolbar_->setToolButtonStyle(Qt::ToolButtonIconOnly);
            main->addToolBar(Qt::TopToolBarArea, scriptsToolbar_);

            QAction* actionToggleScriptsToolBox = new QAction(main);
            actionToggleScriptsToolBox->setObjectName(QString::fromUtf8("actionToggleScriptsToolBox"));
            actionToggleScriptsToolBox->setCheckable(true);
            main->getItems()->menuToolbars->addAction(actionToggleScriptsToolBox);
            actionToggleScriptsToolBox->setText(QApplication::translate("MainWindow", "&Plugin scripts", 0, QApplication::UnicodeUTF8));
            actionToggleScriptsToolBox->setToolTip(QApplication::translate("MainWindow", "Toggle the plugin scripts toolbox", 0, QApplication::UnicodeUTF8));

            connect(actionToggleScriptsToolBox, SIGNAL(toggled(bool)), scriptsToolbar_, SLOT(setVisible(bool)));
            connect((Support::ToolBar*)scriptsToolbar_.data(), SIGNAL(visibleChanged(bool)), actionToggleScriptsToolBox, SLOT(setChecked(bool)));
        }

        QString iconpath = read->iconpath().c_str();
        QFileInfo iconinfo(iconpath);
        if (!iconinfo.exists())
        {
            if (iconinfo.isRelative())
                iconpath = info.path() + QDir::separator() + iconpath;
        }

        TaskInfo("Creating action %s with icon %s", actionText.toLatin1().data(), iconpath.toLatin1().data());

        action = scriptsToolbar_->addAction( QIcon(iconpath), actionText );
    }

    if (!action)
    {
        TaskInfo("Creating action %s", actionText.toLatin1().data());
        action = new QAction(actionText, scripts_ );
    }

    action->setData( info.absoluteFilePath());
    scripts_->addAction( action );
    connect( action, SIGNAL(triggered()), SLOT(createFromScriptPath()));
}


void MatlabController::
        createFromScriptPath()
{
    QAction* a = dynamic_cast<QAction*>(sender());
    BOOST_ASSERT( a );

    TaskInfo ti("createFromScriptPath %s", a->data().toString().toStdString().c_str() );
    Adapters::ReadMatlabSettings::readSettingsAsync( a->data().toString(), this, SLOT(createFromDefaultSettings(Adapters::DefaultMatlabFunctionSettings)));
}


void MatlabController::
        createFromDefaultSettings( Adapters::DefaultMatlabFunctionSettings settings )
{
    createFromSettings( settings );
}


void MatlabController::
        createFromSettings( Adapters::MatlabFunctionSettings& settings )
{
    TaskInfo ti("createFromSettings %s", settings.scriptname().c_str() );
    // find out if this is a source or not
    Adapters::ReadMatlabSettings* testSource = new Adapters::ReadMatlabSettings( settings.scriptname().c_str(), Adapters::ReadMatlabSettings::MetaData_Source );
    testSource->settings = settings;
    connect( testSource, SIGNAL(sourceRead()), SLOT(scriptIsSource()), Qt::DirectConnection);
    connect( testSource, SIGNAL(failed(QString,QString)), SLOT(scriptIsNotSource(QString, QString)), Qt::DirectConnection);
    testSource->readAsyncAndDeleteSelfWhenDone();
}


void MatlabController::
        scriptIsSource()
{
    Adapters::ReadMatlabSettings* read = dynamic_cast<Adapters::ReadMatlabSettings*>(sender());
    BOOST_ASSERT( read );

    TaskInfo ti("scriptIsSource %s", read->settings.scriptname().c_str() );
    if (read->sourceBuffer())
    {
        Signal::pOperation o( new Signal::BufferSource(read->sourceBuffer()));
        Signal::pOperation s( new Signal::OperationSuperposition( Signal::pOperation(), o ));
        ((Signal::OperationSuperposition*)s.get())->name( o->name() );

        project_->appendOperation( s );
        s->invalidate_samples(Signal::Interval::Interval_ALL);

        updateStoredSettings( &read->settings );
    }
    else
    {
        read->settings.setAsSource();
        showNewMatlabOperationDialog( &read->settings );
    }
}


void MatlabController::
        scriptIsNotSource(QString filename, QString info)
{
    if (!info.isEmpty())
    {
        TaskInfo ti("Couldn't determine if script is a source script: %s\n%s", filename.toStdString().c_str(), info.toStdString().c_str() );

        QMessageBox message(
                QMessageBox::Information,
                "Couldn't run script",
                QString("Couldn't run script \"%1\". See details on error below.").arg(filename));

        message.setDetailedText( info );

        message.exec();
        return;
    }

    Adapters::ReadMatlabSettings* read = dynamic_cast<Adapters::ReadMatlabSettings*>(sender());
    BOOST_ASSERT( read );

    TaskInfo ti("scriptIsNotSource %s", read->settings.scriptname().c_str() );
    TaskInfo("Filename: %s", filename.toStdString().c_str());
    TaskInfo("Info: %s", info.toStdString().c_str());
    Adapters::DefaultMatlabFunctionSettings* settings = &read->settings;

    MatlabOperationWidget* settingswidget = new MatlabOperationWidget( settings, project_ );
    createOperation( settingswidget );
}


void MatlabController::
        createFromSettingsFailed( QString filename, QString info )
{
    TaskInfo ti("createFromSettingsFailed %s\n%s", filename.toStdString().c_str(), info.toStdString().c_str() );

    QMessageBox message(
            QMessageBox::Information,
            "Couldn't run script",
            QString("Couldn't run script \"%1\". See details on error below.").arg(filename));

    message.setDetailedText( info );

    message.exec();
}


void MatlabController::
        createView()
{
    foreach(Signal::pChain c, project_->layers.layers())
    {
        createView(c->root_source().get());
    }
}


void MatlabController::
        createView(Signal::Operation* o)
{
    Adapters::MatlabOperation* operation = dynamic_cast<Adapters::MatlabOperation*>(o);
    if (operation)
    {
        MatlabOperationWidget* settings = new MatlabOperationWidget( operation->settings(), project_ );
        operation->settings( settings );

        Signal::pOperation om;
        foreach(Signal::Operation* c, operation->outputs())
        {
            BOOST_ASSERT(c->source().get() == operation);
            om = c->source();
        }
        if (om)
            connectOperation(settings, om);
    }

    // work recursively up to find all operations
    foreach(Signal::Operation* p, o->outputs())
    {
        // verify a correct structure while at it
        BOOST_ASSERT( p->Operation::source().get() == o );

        createView( p );
    }
}


void MatlabController::
        receiveMatlabOperation()
{
    showNewMatlabOperationDialog(0);
}


void MatlabController::
        showNewMatlabOperationDialog(Adapters::MatlabFunctionSettings* psettings)
{
    /*if (_matlaboperation)
    {
        // Already created, make it re-read the script
        dynamic_cast<Adapters::MatlabOperation*>(_matlaboperation.get())->restart();
        worker_->invalidate_post_sink(_matlaboperation->affected_samples());
    }
    else*/
    {
        QDialog d( project_->mainWindow() );
        d.setWindowTitle("Create Matlab/Octave operation");
        d.setLayout( new QVBoxLayout );
        d.layout()->setMargin(0);
        MatlabOperationWidget* settings = new MatlabOperationWidget( psettings, project_ );
        d.layout()->addWidget( settings );
        QDialogButtonBox* buttonBox = new QDialogButtonBox;
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(30, 460, 471, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
        buttonBox->raise();
        d.connect(buttonBox, SIGNAL(accepted()), SLOT(accept()));
        d.connect(buttonBox, SIGNAL(rejected()), SLOT(reject()));

        QWidget* l = new QWidget( &d ); // add default margins around buttonbox
        l->setLayout( new QVBoxLayout );
        l->layout()->addWidget( buttonBox );
        d.layout()->addWidget( l );
        d.hide();
        d.setWindowModality( Qt::WindowModal );
        if (QDialog::Accepted == d.exec())
        {
            if (settings->scriptname().empty())
            {
                // Open terminal
                createOperation( settings );
                updateScriptsMenu();
            }
            else
            {
                bool success = true;
                QFileInfo fi( settings->scriptname().c_str() );
                if (!fi.exists())
                {
                    QMessageBox::warning( project_->mainWindow(), "Opening file", QString("Cannot open file '%1'!").arg(settings->scriptname().c_str()) );
                    success = false;
                }

                QString pattern = "[a-z][a-z0-9]*\\.m";
                if (!QRegExp(pattern, Qt::CaseInsensitive).exactMatch( fi.fileName()))
                {
                    QMessageBox::warning( project_->mainWindow(), "Starting script", "The filename '" + fi.fileName() + "' of a script must match the pattern " + pattern + ". Can't start script.");
                    success = false;
                }

                if (success)
                {
                    createFromSettings( *settings );
                    updateScriptsMenu();
                }
            }
        }
    }

    render_view_->userinput_update();
    project_->setModified();
}


void MatlabController::
        createOperation(MatlabOperationWidget* settings)
{
    Adapters::MatlabOperation* m = new Adapters::MatlabOperation(project_->head->head_source(), settings);
    Signal::pOperation matlaboperation(m);
    connectOperation(settings, matlaboperation);

    bool noscript = settings->scriptname().empty();

    if (!noscript)
        project_->appendOperation( matlaboperation );
}


void MatlabController::
        connectOperation(MatlabOperationWidget* settings, Signal::pOperation matlaboperation)
{
    Adapters::MatlabOperation* m = dynamic_cast<Adapters::MatlabOperation*>(matlaboperation.get());

    if (!settings->hasProcess())
    {
        if (settings->scriptname().empty())
            QErrorMessage::qtHandler()->showMessage("Couldn't start Octave console, make sure that the installation folder is added to your path");
        else
            QErrorMessage::qtHandler()->showMessage("Couldn't start neither Octave nor Matlab, make sure that the installation folder is added to your path");

        return;
    }

    settings->setParent(0);
    connect( render_view_, SIGNAL(populateTodoList()), settings, SLOT(populateTodoList()));
    bool noscript = settings->scriptname().empty();

    if (noscript)
    {
        settings->showOutput();
        settings->ownOperation = matlaboperation;
    }
    else
    {
        m->invalidate_samples( Signal::Intervals::Intervals_ALL );
        m->plotlines.reset( new Tools::Support::PlotLines( render_view_->model ));
        connect( render_view_, SIGNAL(painting()), m->plotlines.get(), SLOT(draw()) );

        updateStoredSettings( settings );
    }
}


void MatlabController::
        updateStoredSettings(Adapters::MatlabFunctionSettings* settings)
{
    QString basename = QFileInfo(settings->scriptname().c_str()).baseName();
    QSettings state;
    state.beginGroup("MatlabOperation");
    state.beginGroup( basename );
    state.setValue("path", QString::fromStdString( settings->scriptname()) );
    state.setValue("chunksize", settings->chunksize() );
    state.setValue("computeInOrder", settings->computeInOrder() );
    state.setValue("redundant", settings->redundant() );
    state.setValue("arguments", QString::fromStdString( settings->arguments()) );
    state.endGroup();
    state.endGroup();
}


void MatlabController::
        receiveMatlabFilter()
{
    /*if (_matlabfilter)
    {
        // Already created, make it re-read the script
        dynamic_cast<Adapters::MatlabFilter*>(_matlabfilter.get())->restart();
        worker_->invalidate_post_sink(_matlabfilter->affected_samples());
    }
    else*/
    {
        Signal::pOperation matlabfilter( new Adapters::MatlabFilter( "matlabfilter" ) );
        project_->appendOperation( matlabfilter );

#ifndef SAWE_NO_MUTEX
        // Make sure the worker runs in a separate thread
        project_->worker_->start();
#endif
    }

    render_view_->userinput_update();
    project_->setModified();
}


void MatlabController::
        tryHeadAsMatlabOperation()
{
    Signal::pOperation t = project_->head->head_source();
    while (true)
    {
        if (dynamic_cast<Signal::OperationCacheLayer*>(t.get()))
            t = t->source();
        else if (dynamic_cast<Tools::Support::OperationOnSelection*>(t.get()))
            t = dynamic_cast<Tools::Support::OperationOnSelection*>(t.get())->operation();
        else
            break;
    }

    Adapters::MatlabOperation* m = dynamic_cast<Adapters::MatlabOperation*>(t.get());

    if (m)
    {
        QDockWidget* toolWindow = project_->mainWindow()->getItems()->toolPropertiesWindow;
        MatlabOperationWidget* w = dynamic_cast<MatlabOperationWidget*>( m->settings() );
        BOOST_ASSERT( w );
        toolWindow->setWidget( w );
        if (w->getOctaveWindow())
        {
            w->getOctaveWindow()->isVisible();
            w->getOctaveWindow()->raise();
        }
        //toolWindow->hide();
        //toolWindow->show();
        //toolWindow->raise();
    }
}


} // namespace Tools
