#include "matlabcontroller.h"
#include "matlaboperationwidget.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

#include "heightmap/collection.h"
#include "tfr/cwt.h"

#include <QDialogButtonBox>
#include <QFile>
#include <QSharedPointer>

namespace Tools {


MatlabController::
        MatlabController( Sawe::Project* project, RenderView* render_view )
            :
            project_(project),
            render_view_(render_view)
{
    setupGui(project);
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
                settings->operation = m;
                if (settings->scriptname().empty())
                {
                    settings->showOutput();
                    settings->ownOperation = matlaboperation;
                }
                else
                {
                    m->invalidate_samples( Signal::Interval(0, project_->head->head_source()->number_of_samples()) );
                    project_->head->appendOperation( matlaboperation );
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
        project_->head->appendOperation( matlabfilter );

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

    Adapters::MatlabOperation* m = dynamic_cast<Adapters::MatlabOperation*>(t.get());

    if (m)
    {
        QDockWidget* toolWindow = project_->mainWindow()->getItems()->toolPropertiesWindow;
        MatlabOperationWidget* w = dynamic_cast<MatlabOperationWidget*>( m->settings() );
        toolWindow->setWidget( w );
        //toolWindow->hide();
        toolWindow->show();
        toolWindow->raise();
    }
}


} // namespace Tools
