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

namespace Tools {


MatlabController::
        MatlabController( Sawe::Project* project, RenderView* render_view )
            :
            project_(project),
            render_view_(render_view)
{
    setupGui(project);

    // Clean up old h5 files that were probably left from a previous crash
    // if no other project is currently running
    // (note, other instances of Sonic AWE might still be running)
    if ( 0==Sawe::Application::global_ptr()->count_projects())
    {
        QDateTime now = QDateTime::currentDateTime();
        foreach (QFileInfo s, QDir::current().entryInfoList( QStringList("*.0x*.h5") ))
        {
            if (QRegExp(".*\\.0x[0-9a-f]{6,16}\\.h5").exactMatch(s.fileName()))
            {
                // Delete it if it was created more than 15 minutes ago
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
}


MatlabController::
        ~MatlabController()
{
    TaskInfo("~MatlabController");
}


void MatlabController::
        setupGui(Sawe::Project* project)
{
    ::Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(ui->actionToogleMatlabToolBox, SIGNAL(toggled(bool)), ui->toolBarMatlab, SLOT(setVisible(bool)));
    connect(ui->toolBarMatlab, SIGNAL(visibleChanged(bool)), ui->actionToogleMatlabToolBox, SLOT(setChecked(bool)));

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

    connect( project->head.get(), SIGNAL(headChanged()), SLOT(tryHeadAsMatlabOperation()));
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
                Adapters::MatlabOperation* m = new Adapters::MatlabOperation(Signal::pOperation(), settings);
                Signal::pOperation matlaboperation(m);
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
                }
            }
        }
    }

    render_view_->userinput_update();
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
