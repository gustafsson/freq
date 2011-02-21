#include "matlabcontroller.h"
#include "matlaboperationwidget.h"

// Sonic AWE
#include "adapters/matlabfilter.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

#include "heightmap/collection.h"
#include "tfr/cwt.h"

#include <QDialogButtonBox>

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
        MatlabOperationWidget* settings = new MatlabOperationWidget( project_->head->head_source()->sample_rate() );
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
            _matlaboperation = settings->createMatlabOperation();
            if (_matlaboperation)
                project_->head->appendOperation( _matlaboperation );
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
        _matlabfilter.reset( new Adapters::MatlabFilter( "matlabfilter" ) );
        project_->head->appendOperation( _matlabfilter );

#ifndef SAWE_NO_MUTEX
        // Make sure the worker runs in a separate thread
        project_->worker_->start();
#endif
    }

    render_view_->userinput_update();}


} // namespace Tools
