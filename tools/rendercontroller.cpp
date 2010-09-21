#include "rendercontroller.h"
#include "ui/comboboxaction.h"
#include "ui/displaywidget.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

#include <CudaException.h>
#include <cuda.h>

using namespace Ui;

namespace Tools
{

RenderController::
        RenderController( RenderView *view )
            :
            QWidget(view->model->project->mainWindow()),
            model(view->model),
            view(view)
{
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model->project->mainWindow());
    QToolBar* toolBarRender = new QToolBar(main);
    toolBarRender->setObjectName(QString::fromUtf8("toolBarTool"));
    toolBarRender->setEnabled(true);
    toolBarRender->setContextMenuPolicy(Qt::NoContextMenu);
    toolBarRender->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::BottomToolBarArea, toolBarRender);


    {   ComboBoxAction * qb = new ComboBoxAction();
        qb->addActionItem( main->ui->actionToggle_hz_grid );
        qb->addActionItem( main->ui->actionToggle_piano_grid );
        toolBarRender->addWidget( qb );
    }

    {   ComboBoxAction * qb = new ComboBoxAction();
        qb->decheckable( false );
        qb->addActionItem( main->ui->actionSet_rainbow_colors );
        qb->addActionItem( main->ui->actionSet_grayscale );
        toolBarRender->addWidget( qb );
    }

    {   ComboBoxAction * qb = new ComboBoxAction();
        qb->addActionItem( main->ui->actionTransform_Cwt );
        qb->addActionItem( main->ui->actionTransform_Stft );
        qb->addActionItem( main->ui->actionTransform_Cwt_phase );
        qb->addActionItem( main->ui->actionTransform_Cwt_reassign );
        qb->addActionItem( main->ui->actionTransform_Cwt_ridge );
        qb->decheckable( false );
        toolBarRender->addWidget( qb );
    }

    QLayout* layout = new QVBoxLayout();
    this->setLayout( layout );

    Ui::DisplayWidget* d = new Ui::DisplayWidget( model->project, this, model );
    layout->addWidget( d );
    view->displayWidget = d;

    float l = model->project->worker.source()->length();
    view->setPosition( std::min(l, 10.f)*0.5f, 0.5f );


    main->setCentralWidget(this);

    hide();
    show();

    // TODO remove
    // connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateLayerList(Signal::pOperation)));
    // connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateOperationsTree(Signal::pOperation)));
    //connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
    //connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));

    connect(main->ui->actionActivateSelection, SIGNAL(toggled(bool)), d, SLOT(receiveToggleSelection(bool)));
    connect(main->ui->actionActivateNavigation, SIGNAL(toggled(bool)), d, SLOT(receiveToggleNavigation(bool)));
    connect(main->ui->actionActivateInfoTool, SIGNAL(toggled(bool)), d, SLOT(receiveToggleInfoTool(bool)));
    connect(main->ui->actionPlaySelection, SIGNAL(triggered()), d, SLOT(receivePlaySound()));
    connect(main->ui->actionFollowPlayMarker, SIGNAL(triggered(bool)), d, SLOT(receiveFollowPlayMarker(bool)));
    connect(main->ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), d, SLOT(receiveTogglePiano(bool)));
    connect(main->ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), d, SLOT(receiveToggleHz(bool)));
    connect(main->ui->actionActionAdd_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddSelection(bool)));
    connect(main->ui->actionActionRemove_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddClearSelection(bool)));
    connect(main->ui->actionCropSelection, SIGNAL(triggered()), d, SLOT(receiveCropSelection()));
    connect(main->ui->actionMoveSelection, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelection(bool)));
    connect(main->ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelectionInTime(bool)));
    connect(main->ui->actionMatlabOperation, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabOperation(bool)));
    connect(main->ui->actionMatlabFilter, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabFilter(bool)));
    connect(main->ui->actionTonalizeFilter, SIGNAL(triggered(bool)), d, SLOT(receiveTonalizeFilter(bool)));
    connect(main->ui->actionReassignFilter, SIGNAL(triggered(bool)), d, SLOT(receiveReassignFilter(bool)));
    connect(main->ui->actionRecord, SIGNAL(triggered(bool)), d, SLOT(receiveRecord(bool)));
    connect(d, SIGNAL(setSelectionActive(bool)), main->ui->actionActivateSelection, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setNavigationActive(bool)), main->ui->actionActivateNavigation, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setInfoToolActive(bool)), main->ui->actionActivateInfoTool, SLOT(setChecked(bool)));
    connect(main->ui->actionSet_rainbow_colors, SIGNAL(triggered()), d, SLOT(receiveSetRainbowColors()));
    connect(main->ui->actionSet_grayscale, SIGNAL(triggered()), d, SLOT(receiveSetGrayscaleColors()));
    connect(main->ui->actionSet_heightlines, SIGNAL(toggled(bool)), d, SLOT(receiveSetHeightlines(bool)));

    connect(main->ui->actionTransform_Cwt, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt()));
    connect(main->ui->actionTransform_Stft, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Stft()));
    connect(main->ui->actionTransform_Cwt_phase, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_phase()));
    connect(main->ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_reassign()));
    connect(main->ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_ridge()));

    {   QSlider * qs = new QSlider();
        qs->setOrientation( Qt::Horizontal );
        qs->setValue( 50 );
        qs->setToolTip( "Intensity level" );
        connect(qs, SIGNAL(valueChanged(int)), d, SLOT(receiveSetYScale(int)));

        main->ui->toolBarPlay->addWidget( qs );
    }

    {   QSlider * qs = new QSlider();
        qs->setOrientation( Qt::Horizontal );
        qs->setValue( 50 );
        qs->setToolTip( "Time/frequency resolution. If set higher than the middle, the audio reconstruction will be incorrect." );
        connect(qs, SIGNAL(valueChanged(int)), d, SLOT(receiveSetTimeFrequencyResolution(int)));

        main->ui->toolBarPlay->addWidget( qs );
    }


    main->ui->actionActivateNavigation->setChecked(true);

    // updateOperationsTree( project->worker.source() );
    //d->getCwtFilterHead();

    if (d->isRecordSource()) {
        main->ui->actionRecord->setEnabled(true);
    } else {
        main->ui->actionRecord->setEnabled(false);
    }
}


RenderController::
        ~RenderController()
{
    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    model->collection->reset();

    // Because the cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
    BOOST_ASSERT( QGLContext::currentContext() );

    // Destroy the cuda context for this thread
    CudaException_SAFE_CALL( cudaThreadExit() );
}


} // namespace Tools
