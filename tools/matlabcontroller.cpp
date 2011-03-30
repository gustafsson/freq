#include "matlabcontroller.h"
#include "matlaboperationwidget.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "support/operation-composite.h"

#include "heightmap/collection.h"
#include "tfr/cwt.h"
#include "sawe/application.h"

#include <QDialogButtonBox>
#include <QFile>
#include <QSharedPointer>
#include <QDir>
#include <QRegExp>
#include <QDateTime>
#include <QSettings>
#include <QErrorMessage>

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
    foreach (QFileInfo s, QDir::current().entryInfoList( QStringList("*.0x*.h5") ))
    {
        if (QRegExp(".*\\.0x[0-9a-f]{6,16}\\.h5").exactMatch(s.fileName()) ||
            QRegExp(".*\\.0x[0-9a-f]{6,16}\\.h5.result.h5").exactMatch(s.fileName()))
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
    MatlabOperationWidget* settings = new MatlabOperationWidget( project_ );
    settings->scriptname( m->settings()->scriptname() );
    settings->redundant( m->settings()->redundant() );
    settings->computeInOrder( m->settings()->computeInOrder() );
    settings->chunksize( m->settings()->chunksize() );
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
        scripts_ = new QMenu( "Matlab/Octave &scripts",  ui->menuWindows );
        ui->menuWindows->insertMenu( ui->menuToolbars->menuAction(), scripts_ );
    }
    scripts_->clear();
    scripts_->insertAction( 0, ui->actionMatlabOperation );

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

            QAction* action = new QAction(g, scripts_ );
            scripts_->addAction( action );
            connect( action, SIGNAL(triggered()), SLOT(createFromAction()));
        }
    }
    state.endGroup();
}


void MatlabController::
        createFromAction()
{
    QAction* a = dynamic_cast<QAction*>(sender());
    BOOST_ASSERT( a );

    QSettings state;
    state.beginGroup("MatlabOperation");
    state.beginGroup( a->text());

    MatlabOperationWidget* settings = new MatlabOperationWidget( project_ );
    settings->scriptname( state.value("path").toString().toStdString() );
    settings->chunksize( state.value("chunksize").toInt() );
    settings->computeInOrder( state.value("computeInOrder").toBool() );
    settings->redundant( state.value("redundant").toInt() );
    settings->arguments( state.value("arguments").toString().toStdString() );
    state.endGroup();
    state.endGroup();

    createOperation( settings );
}


void MatlabController::
        receiveMatlabOperation()
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
        d.setWindowTitle("Create Matlab operation");
        d.setLayout( new QVBoxLayout );
        MatlabOperationWidget* settings = new MatlabOperationWidget( project_ );
        d.layout()->addWidget( settings );
        QDialogButtonBox* buttonBox = new QDialogButtonBox;
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(30, 460, 471, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
        buttonBox->raise();
        d.connect(buttonBox, SIGNAL(accepted()), SLOT(accept()));
        d.connect(buttonBox, SIGNAL(rejected()), SLOT(reject()));

        d.layout()->addWidget( buttonBox );
        d.hide();
        d.setWindowModality( Qt::WindowModal );
        if (QDialog::Accepted == d.exec())
        {
            if (!settings->scriptname().empty() && !QFile::exists( settings->scriptname().c_str() ))
            {
                QMessageBox::warning( project_->mainWindow(), "Opening file", QString("Cannot open file '%1'!").arg(settings->scriptname().c_str()) );
            }
            else
            {
                createOperation( settings );
                updateScriptsMenu();
            }
        }
    }

    render_view_->userinput_update();
}


void MatlabController::
        createOperation(MatlabOperationWidget* settings)
{
    Adapters::MatlabOperation* m = new Adapters::MatlabOperation(project_->head->head_source(), settings);
    Signal::pOperation matlaboperation(m);
    if (!settings->hasProcess())
    {
        QErrorMessage::qtHandler()->showMessage("Couldn't start neither Octave nor Matlab, make sure that the installation folder is added to your path");
        return;
    }

    settings->setParent(0);
    connect( render_view_, SIGNAL(populateTodoList()), settings, SLOT(populateTodoList()));
    bool noscript = settings->scriptname().empty();
    settings->setOperation( m );
    if (noscript)
    {
        settings->showOutput();
        settings->ownOperation = matlaboperation;
    }
    else
    {
        m->invalidate_samples( Signal::Intervals::Intervals_ALL );
        project_->appendOperation( matlaboperation );
        m->plotlines.reset( new Tools::Support::PlotLines( render_view_->model ));
        connect( render_view_, SIGNAL(painting()), m->plotlines.get(), SLOT(draw()) );

        QSettings state;
        state.beginGroup("MatlabOperation");
        state.beginGroup( QString::fromStdString( m->functionName() ) );
        state.setValue("path", QString::fromStdString( m->settings()->scriptname()) );
        state.setValue("chunksize", m->settings()->chunksize() );
        state.setValue("computeInOrder", m->settings()->computeInOrder() );
        state.setValue("redundant", m->settings()->redundant() );
        state.setValue("arguments", QString::fromStdString( m->settings()->arguments()) );
        state.endGroup();
        state.endGroup();
    }
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
}


void MatlabController::
        tryHeadAsMatlabOperation()
{
    Signal::pOperation t = project_->head->head_source();
    if (dynamic_cast<Signal::OperationCacheLayer*>(t.get()))
        t = t->source();
    if (dynamic_cast<Tools::Support::OperationOnSelection*>(t.get()))
        t = dynamic_cast<Tools::Support::OperationOnSelection*>(t.get())->selection();

    Adapters::MatlabOperation* m = dynamic_cast<Adapters::MatlabOperation*>(t.get());

    if (m)
    {
        QDockWidget* toolWindow = project_->mainWindow()->getItems()->toolPropertiesWindow;
        MatlabOperationWidget* w = dynamic_cast<MatlabOperationWidget*>( m->settings() );
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
